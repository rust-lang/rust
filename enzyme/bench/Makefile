.SUFFIXES:

.PHONY : clean

LOADSO=-load=../build-dbg/Enzyme/LLVMEnzyme-7.so 
CLANG=../../build/bin/clang
OPT=../../build/bin/opt

%-unopt.ll: %.c
	#$(CLANG) $^ -O2 -S -emit-llvm -o - -o $@ 
	#$(CLANG) $^ -O3 -S -emit-llvm -o - -o $@ 
	#$(CLANG) -fno-unroll-loops $^ -O1 -Xclang -disable-llvm-passes -fno-vectorize -fno-slp-vectorize -S -emit-llvm -o - -ffast-math -o $@ 
	$(CLANG) -fno-unroll-loops $^ -O1 -fno-vectorize -fno-slp-vectorize -S -emit-llvm -o - -ffast-math -o $@ 

%-unopt.ll: %.cpp
	#$(CLANG)++ -fno-exceptions -fno-use-cxa-atexit -fno-unroll-loops $^ -O0 -fno-vectorize -fno-slp-vectorize -S -emit-llvm -o $@ -ffast-math -Wno-error=non-pod-varargs -DEIGEN_UNROLLING_LIMIT=0 -I/home/wmoses/autodiff/adept-2.0.5/include #-ggdb 
	$(CLANG)++ -std=c++2a -fno-exceptions -fno-use-cxa-atexit -fno-unroll-loops $^ -O1 -fno-vectorize -fno-slp-vectorize -S -emit-llvm -o $@ -ffast-math -Wno-error=non-pod-varargs -DEIGEN_UNROLLING_LIMIT=0 -I/home/wmoses/autodiff/adept-2.0.5/include #-ggdb 
	#$(CLANG)++ -fno-exceptions -fno-use-cxa-atexit -fno-unroll-loops $^ -O3 -fno-vectorize -fno-slp-vectorize -S -emit-llvm -o $@ -ffast-math -Wno-error=non-pod-varargs -DEIGEN_UNROLLING_LIMIT=0 -I/home/wmoses/autodiff/adept-2.0.5/include #-ggdb 
	#$(CLANG)++ -fno-exceptions -fno-use-cxa-atexit -fno-unroll-loops $^ -O3 -fno-vectorize -fno-slp-vectorize -S -emit-llvm -o $@ -ffast-math -Wno-error=non-pod-varargs -DEIGEN_UNROLLING_LIMIT=0 -I/home/wmoses/autodiff/adept-2.0.5/include -march=atom #-ggdb 
	#$(CLANG)++ -fno-exceptions -fno-unroll-loops $^ -O3 -fno-vectorize -fno-slp-vectorize -S -emit-llvm -o $@ -ffast-math -Wno-error=non-pod-varargs -DEIGEN_UNROLLING_LIMIT=0 -I/home/wmoses/autodiff/adept-2.0.5/include -ggdb 

%-preopt.ll: %-unopt.ll
	$(OPT) $^ -indvars $(LOADSO) -enzyme -mem2reg -sroa -early-cse-memssa -adce -bdce -simplifycfg -inline -adce -aggressive-instcombine -O2 -S -o $@

%-raw.ll: %-unopt.ll
	$(OPT) $^ -indvars $(LOADSO) -enzyme -S -o $@

%-raw2.ll: %-unopt.ll
	$(OPT) $^ -indvars $(LOADSO) -enzyme -simplifycfg -mem2reg -sroa -S -o $@
	#../build/bin/opt $^ -indvars -lower-autodiff -simplifycfg -mem2reg -sroa -adce -bdce -aggressive-instcombine -S -o $@
	#../build/bin/opt $^ -indvars -lower-autodiff -S -o - | ../build/bin/opt - -S -o $@ -adce -bdce -aggressive-instcombine -instsimplify -correlated-propagation -early-cse-memssa -adce -bdce -aggressive-instcombine -simplifycfg -loop-deletion -simplifycfg -correlated-propagation -deadargelim -adce -bdce -aggressive-instcombine -inline -O1
	#../build/bin/opt $^ -indvars -lower-autodiff -mem2reg -sroa -adce -bdce -aggressive-instcombine -S -o - | ../build/bin/opt - -S -o $@ -instsimplify -correlated-propagation -early-cse-memssa -adce -bdce -aggressive-instcombine -simplifycfg -loop-deletion -simplifycfg -correlated-propagation -adce  

%-lowernovec.ll: %-unopt.ll
	#../build/bin/opt $^ -indvars -lower-autodiff -mem2reg -sroa -adce -bdce -aggressive-instcombine -S -o $@ -instsimplify -early-cse-memssa -adce -bdce -aggressive-instcombine -simplifycfg -loop-deletion -simplifycfg -correlated-propagation -adce -inline -instsimplify -instcombine -correlated-propagation
	$(OPT) $^ -indvars $(LOADSO) -enzyme -mem2reg -sroa -adce -bdce -aggressive-instcombine -S -o $@ -instsimplify -early-cse-memssa -adce -bdce -aggressive-instcombine -simplifycfg -loop-deletion -simplifycfg -correlated-propagation -adce -inline -instsimplify -instcombine -correlated-propagation #-autodiff_optimize=true

%-lowerprint.ll: %-unopt.ll
	$(OPT) $^ -indvars $(LOADSO) -enzyme -verify -mem2reg -sroa -adce -bdce -aggressive-instcombine -S -o $@ -instsimplify -correlated-propagation -early-cse-memssa -adce -bdce -aggressive-instcombine -simplifycfg -loop-deletion -simplifycfg -correlated-propagation -deadargelim -adce -bdce -aggressive-instcombine -inline -O3 -print-before-all -print-after-all

%-lowerq.ll: %-unopt.ll
	$(OPT) $^ -S -indvars $(LOADSO) -enzyme -enzyme_inline=0 -verify -mem2reg -enzyme_print=0 -enzyme_nonmarkedglobals_inactive=1 -enzyme_emptyfnconst=0 -enzyme_printconst=0 -o $@ #| $(OPT) - -S -o $@ -instsimplify -correlated-propagation -early-cse-memssa -adce -bdce -aggressive-instcombine -simplifycfg -loop-deletion -simplifycfg -correlated-propagation -deadargelim -adce -bdce -aggressive-instcombine #-inline -O3

%-lower.ll: %-unopt.ll
	#$(OPT) $^ -ipconstprop -always-inline -mem2reg -indvars $(LOADSO) -enzyme -enzyme_inline=1 -verify -early-cse -S -o - -enzyme_print=0 -enzyme_nonmarkedglobals_inactive=0 -enzyme_emptyfnconst=1 -enzyme_printconst=0 | $(OPT) - -S -o $@ 
	$(OPT) $^ -ipconstprop -always-inline -mem2reg -indvars $(LOADSO) -enzyme -enzyme_inline=0 -verify -mem2reg -sroa -adce -bdce -aggressive-instcombine -S -o - -enzyme_print=0 -enzyme_nonmarkedglobals_inactive=1 -enzyme_emptyfnconst=0 -enzyme_printconst=0 | $(OPT) - -S -o $@ -instsimplify -correlated-propagation -early-cse-memssa -adce -bdce -aggressive-instcombine -simplifycfg -loop-deletion -simplifycfg -correlated-propagation -deadargelim -adce -bdce -aggressive-instcombine #-inline -O3
	#$(OPT) $^ -ipconstprop -always-inline -mem2reg -indvars $(LOADSO) -enzyme -enzyme_inline=1 -verify -mem2reg -sroa -adce -bdce -aggressive-instcombine -S -o - -enzyme_print=0 -enzyme_nonmarkedglobals_inactive=0 -enzyme_emptyfnconst=1 -enzyme_printconst=0 | $(OPT) - -S -o $@ -instsimplify -correlated-propagation -early-cse-memssa -adce -bdce -aggressive-instcombine -simplifycfg -loop-deletion -simplifycfg -correlated-propagation -deadargelim -adce -bdce -aggressive-instcombine #-inline -O3
	#../build/bin/opt $^ -indvars -autodiff_inline=true -lower-autodiff -verify -mem2reg -sroa -adce -bdce -aggressive-instcombine -S -o - | ../build/bin/opt - -S -o $@ -instsimplify -correlated-propagation -early-cse-memssa -adce -bdce -aggressive-instcombine -simplifycfg -loop-deletion -simplifycfg -correlated-propagation -deadargelim -adce -bdce -aggressive-instcombine -inline -O3
	#../build/bin/opt $^ -indvars -lower-autodiff -mem2reg -sroa -early-cse-memssa -adce -bdce -aggressive-instcombine -simplifycfg -correlated-propagation -deadargelim -adce -bdce -aggressive-instcombine -O3 -S -o $@

%.o: %-lower.ll
	$(CLANG) $^ -o $@ -lm -lstdc++ -g -ggdb #-lblas #../../tapenade/ADFirstAidKit/*.o
	#../build/bin/clang $^ -o $@ -lm -ladept -lstdc++ -g -ggdb -lblas

clean:
	rm -f *.ll *.o

