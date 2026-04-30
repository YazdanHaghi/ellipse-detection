for c++ file:

to build:

g++ -O3 -std=c++17 ellipsesdetetction_visual_fixed.cpp -o dght_cpp -I/usr/include/eigen3 $(pkg-config --cflags --libs opencv4)

and run:

./dght_cpp --ann annotations.json --ellipses_dir Ellipses --mode first --n 1000 --vis_dir vis_output --save_vis_n 20 --out_csv batch_results_all_1000.csv
