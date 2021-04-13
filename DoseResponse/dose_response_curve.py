# Python Dependencies
import pandas as pd
import numpy as np
import seaborn as sns

from DoseResponse.equations import Equations
#from equations import Equations

from inspect import getfullargspec
from scipy.optimize import curve_fit
from scipy.stats.distributions import t
from matplotlib import pyplot as plt


class DoseResponseCurve(object):
    """
        Process raw dose response data, removing baseline or various
        normalizing techniques, and plot using Seaborn + matplotlib libraries.

        Args
            datafile (path)
                filepath to raw data
            method (method)
            top_data (path)
            bottom_data (path)

        Instance Attributes
            top (float)
                Definition of 100% Signal, derived from the mean of top_data used for data normalization

            bottom (float)
                Definition of 0% Signal, derived from the mean of bottom_data used for data normalization

            df_main (DataFrame)
                Initial Raw Data derived from datafile

            df_normalized (DataFrame)
                Normalized Data derived from df_main

            df_summary (DataFrame)
                Summmary Statistics derived from df_main

            n_replicates (int)
                Number of Sample Replicates

            n_compounds (int)
                Number ofIndependent Compounds

            plot (seaborn plot)
                The seaborn plot

            fit_parameters (dict --> DataFrame)
                The Results of Curve Fitting derived from the normalized data
    """
    
    def __init__(self, datafile, 
                 method=None,
                 top_data=None,
                 bottom_data=None):
        
        # Instance Args
        self.datafile = datafile
        self.method = method  

        if top_data is not None:
            self.top = pd.read_csv(top_data, sep='\t', header=None)[0].mean() 
        else:
            self.top = None

        if bottom_data is not None:
            self.bottom = pd.read_csv(bottom_data, sep='\t', header=None)[0].mean()
        else:
            self.bottom = None

        # Instance Attributes
        self.df_main = None
        self.df_normalized = None
        self.df_summary = None
        self.df_plot_ready = None
        self.n_replicates = None
        self.n_compounds = None
        self.plot = None
        self.fit_parameters = {}

    def scatterplot(self, func=None, 
                    xlabel='[Compound] (nM)',
                    ylabel='Anisotropy',
                    palette='viridis_r',
                    baseline_correction=True, 
                    *args, **kargs):
    
        """Plot and Curve Fit data on a log[x] axis."""
        # Initial Checks
        if self.df_main is None:
            self._load_data()
            self._prep_data_for_plotting()
        
        if self.df_plot_ready is None:
            self._prep_data_for_plotting()

        if func is None:
            func = self.method

        # Initializing
        count = 0
        compounds = np.unique(self.df_plot_ready['COMPOUND'])
        df_list = []
        colors = sns.color_palette(palette, self.n_compounds)

        """
            other nice color palettes I like...
                > rainbow
                > ocean_r
                > ocean
                > viridis_r 
        """

        # Iterate Through Compounds in Dataframe and Perform Fit for Each
        for c in compounds:
            # Group Data by Compound and Filter out np.nan values
            df = self.df_plot_ready[(self.df_plot_ready['COMPOUND'] == c) &
                                    (~np.isnan(self.df_plot_ready['value']))].copy()
            
            # Remove Baseline if Data has a Concentration = 0
            # TODO: Provide an Alternative method for an identical baseline across all samples
            # TODO: Provide an Alternative Alternative Method to Group by Sample Date or Experiment ID.
            if baseline_correction:
                baseline = df.loc[df['CONCENTRATION'] == 0, 'value'].mean()
                df['value_corrected'] = df['value'] - baseline
            else:
                df['value_corrected'] = df['value']
            df = df[df['CONCENTRATION'] != 0]

            # Normalize Data by Definition of 100% ...
            if self.top:
                df['value_normalized'] = df['value_corrected'] * 100 / self.top
            else:
                # No Normalization of Data
                df['value_normalized'] = df['value_corrected']
            
            # Add Newly computed Data to List
            df_list.append(df)

            # Fit Curve to Normalized Data
            popt, popv = curve_fit(func,
                                   method='trf',
                                   xdata=df['CONCENTRATION'],
                                   ydata=df['value_normalized'],
                                   xtol=1e-12,
                                   ftol=1e-12,
                                   gtol=1e-12,
                                   *args,
                                   **kargs)

            # Calculate 95% Confidence Intervals
            degrees_of_freedom = max(0, len(df) - len(popt))
            t_value = t.ppf(0.975, degrees_of_freedom)
            l_ci = []
            for val, var in zip(popt, np.diag(popv)):
                sigma = var ** 0.5
                ci = (val - sigma * t_value, val + sigma * t_value)
                l_ci.append(ci)

            # Report Best Fit Values
            self.fit_parameters[c] = [*popt, *l_ci]

            # Add Fitting to Plot
            xdata = np.linspace(start=df['CONCENTRATION'].min(), 
                                stop=df['CONCENTRATION'].max(),
                                num=int(df['CONCENTRATION'].max()),
                                endpoint=True
                                )
            plt.plot(func(xdata, *popt), ':', label=c, color=colors[count])
            count += 1
        
        # Finishing Touches on Dataframe
        df_concat = pd.concat(df_list, axis=0)
        self.df_plot_ready = self.df_plot_ready.merge(df_concat, 
                                                      on=['COMPOUND', 'CONCENTRATION', 'variable', 'value'],
                                                      how='left'
                                                      )
        
        # Finalize Best Fit Reporting
        cols = list(getfullargspec(func))[0][1:]
        columns = [*cols, *[f'{i}_CI' for i in cols]]
        self.fit_parameters = pd.DataFrame.from_dict(self.fit_parameters,
                                                     orient='index',
                                                     columns=columns
                                                     )

        # Prepare Seaborn Scatter plot
        self.plot = sns.scatterplot(
                    data=self.df_plot_ready,
                    hue=self.df_plot_ready['COMPOUND'],
                    x=self.df_plot_ready['CONCENTRATION'],
                    y=self.df_plot_ready['value_normalized'],
                    palette=colors
                                   )  
                          
        # Additional Peripheral Plot Parameters
        self.plot.set(xscale="log",
                      xlabel=xlabel,
                      ylabel=ylabel
                      )
        return self.plot

    def _prep_data_for_plotting(self):
        # Make it easier to sort
        self.df_main[['COMPOUND', 'CONCENTRATION']] = [[i, float(x)] for i, x in self.df_main.index]
        self.n_compounds = len(set(self.df_main['COMPOUND']))
        self.df_plot_ready = pd.melt(self.df_main,
                                     id_vars=['COMPOUND', 'CONCENTRATION'],
                                     value_vars=[n for n in range(self.n_replicates)]
                                     )

    def _remove_baseline(self):
        # Calculate mean of "0% signal"
        if self.bottom is None:
            print('0% Signal is not Defined. Data was not normalized.')
        else:
            baseline = self.bottom.mean()  # no need to define axis of a series
            for n in self.n_replicates:
                self.df_normalized[n] = self.df_main[n] - baseline

    def data_summary(self):
        """This function summarizes the raw Data."""
        # TODO: Create a similar function which performs on normalized
        if self.df_main is None:
            self._load_data()
        self.df_summary = self.df_main.copy()
        self.df_summary['N'] = self.df_main.count(axis=1)
        self.df_summary['MEAN'] = self.df_main.mean(axis=1)
        self.df_summary['SD'] = self.df_main.std(axis=1)

    def _load_data(self):
        """Helper Function to Load data from a file."""
        self.df_main = pd.read_csv(self.datafile, header=[0, 1], sep='\t').T
        self.n_replicates = len(self.df_main.columns)


def main():
    # Manual User Defined Parameters
    filename = 'SampleData/2comp_test.txt'
    output_name = 'Test'

    # Call Class
    x = DoseResponseCurve(
        datafile=filename,
        method=Equations.VariableSlopeDoseResponse
                         )
    
    x.data_summary()
    print(x.df_summary)
    x.scatterplot(func=Equations.VariableSlopeDoseResponse)

    # Output Fitting
    x.df_summary.to_csv(f'{output_name}_summary.txt', sep='\t')
    x.df_main.to_csv(f'{output_name}_output.txt', sep='\t')
    x.df_plot_ready.to_csv(f'{output_name}_plot_ready.txt', sep='\t')
    x.fit_parameters.to_csv(f'{output_name}_Fit_Parameters.txt', sep='\t')

    # Show Dose Response Curves
    plt.show()


if __name__ == "__main__":
    main()
