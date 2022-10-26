path= newArray(1);

path[0]="/Users/Mathew/Documents/Current analysis/Peptide_16_PAINT/SR/"



k=10;
for (i=0; i<path.length; i++){

count = 0;	
n=0;
dir=path[i];
processFiles(dir);

config="/Users/Mathew/gdsc.smlm.settings.xml";		// Admin file. 
prec="50";		// Precision threshold.
sig="20";		// Signal threshold. 


run("Concatenate...", "all_open");
run("Peak Fit", "template=[None] config_file="+config+" calibration=103 gain=38 em-ccd exposure_time=50 initial_stddev0=1.000 initial_stddev1=1.000 initial_angle=0.000 spot_filter_type=Single spot_filter=Mean smoothing=1.20 search_width=1 border=1 fitting_width=3 fit_solver=[Least Squares Estimator (LSE)] fit_function=Circular fail_limit=3 neighbour_height=0.30 residuals_threshold=1 duplicate_distance=0.50 shift_factor=1 signal_strength="+sig+" min_photons=0 min_width_factor=0.50 width_factor=2 precision="+prec+" results_table=Uncalibrated image=[Localisations (width=precision)] image_precision=5 image_scale=8 results_dir=[] results_in_memory camera_bias=0 fit_criteria=[Least-squared error] significant_digits=5 coord_delta=0.0001 lambda=10.0000 max_iterations=20 stack");



saveAs("Tiff", ""+dir+"SuperRes.tif");
close();
close();
		
selectWindow("Fit Results");
saveAs("text",dir+"FitResults.txt");
	
run("Close");
selectWindow("Log");
saveAs("text",dir+"_Details.txt");
run("Close");	


}


/////////// Function to process the files ///////////////

function processFiles(dir) 

	{

	list = getFileList(dir);

		for (i=0; i<list.length; i++)

		 {

		if (!startsWith(list[i],"Log"))

			{

			if (endsWith(list[i], "/"))

		              processFiles(""+dir+list[i]);

         			 else 

			{

		             showProgress(n++, count);

            			path = dir+list[i];

	            		processFile(path);

			}

			}

		}

	}

function processFile(path) 

	{
		       	if (endsWith(path, ".tif") || endsWith(path, "561_1.tif") ) 
 
		{
			open(path);

			
	k++;
 			
//run("Close"); 

		}
	}