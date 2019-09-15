.SUFFIXES:

.PHONY : clean

%-unopt.ll: %.c
	#../build/bin/clang -fno-unroll-loops $^ -O1 -fno-vectorize -fno-slp-vectorize -S -emit-llvm -o - -ffast-math | ../build/bin/opt -early-cse-memssa - -S -o $@ -simplifycfg -simplifycfg
	#../build/bin/clang -fno-unroll-loops $^ -O3 -fno-vectorize -fno-slp-vectorize -S -emit-llvm -o - -ffast-math | ../build/bin/opt -early-cse-memssa - -S -o $@ -simplifycfg -simplifycfg
	../build/bin/clang -fno-unroll-loops $^ -O3 -fno-vectorize -fno-slp-vectorize -S -emit-llvm -o - -ffast-math -o $@ #| ../build/bin/opt -early-cse-memssa - -S -o $@ -simplifycfg -simplifycfg

%-unopt.ll: %.cpp
	#../build/bin/clang++ -fno-unroll-loops $^ -O3 -fno-vectorize -fno-slp-vectorize -S -emit-llvm -o $@ -ffast-math -Wno-error=non-pod-varargs -DEIGEN_UNROLLING_LIMIT=0 -I ../../adept-2.0.5/include -I../../tapenade/ADFirstAidKit
	../build/bin/clang++ -fno-exceptions -fno-unroll-loops $^ -O3 -fno-vectorize -fno-slp-vectorize -S -emit-llvm -o $@ -ffast-math -Wno-error=non-pod-varargs -DEIGEN_UNROLLING_LIMIT=0 -I../../adept-2.0.5/include 

%-preopt.ll: %-unopt.ll
	../build/bin/opt $^ -indvars -load=../enzyme/build/Enzyme/LLVMEnzyme-7.so -enzyme -mem2reg -sroa -early-cse-memssa -adce -bdce -simplifycfg -inline -adce -aggressive-instcombine -O2 -S -o $@

%-raw.ll: %-unopt.ll
	../build/bin/opt $^ -indvars -load=../enzyme/build/Enzyme/LLVMEnzyme-7.so -enzyme -S -o $@

%-raw2.ll: %-unopt.ll
	../build/bin/opt $^ -indvars -load=../enzyme/build/Enzyme/LLVMEnzyme-7.so -enzyme -simplifycfg -mem2reg -sroa -S -o $@
	#../build/bin/opt $^ -indvars -lower-autodiff -simplifycfg -mem2reg -sroa -adce -bdce -aggressive-instcombine -S -o $@
	#../build/bin/opt $^ -indvars -lower-autodiff -S -o - | ../build/bin/opt - -S -o $@ -adce -bdce -aggressive-instcombine -instsimplify -correlated-propagation -early-cse-memssa -adce -bdce -aggressive-instcombine -simplifycfg -loop-deletion -simplifycfg -correlated-propagation -deadargelim -adce -bdce -aggressive-instcombine -inline -O1
	#../build/bin/opt $^ -indvars -lower-autodiff -mem2reg -sroa -adce -bdce -aggressive-instcombine -S -o - | ../build/bin/opt - -S -o $@ -instsimplify -correlated-propagation -early-cse-memssa -adce -bdce -aggressive-instcombine -simplifycfg -loop-deletion -simplifycfg -correlated-propagation -adce  

%-lowernovec.ll: %-unopt.ll
	#../build/bin/opt $^ -indvars -lower-autodiff -mem2reg -sroa -adce -bdce -aggressive-instcombine -S -o $@ -instsimplify -early-cse-memssa -adce -bdce -aggressive-instcombine -simplifycfg -loop-deletion -simplifycfg -correlated-propagation -adce -inline -instsimplify -instcombine -correlated-propagation
	../build/bin/opt $^ -indvars -load=../enzyme/build/Enzyme/LLVMEnzyme-7.so -enzyme -mem2reg -sroa -adce -bdce -aggressive-instcombine -S -o $@ -instsimplify -early-cse-memssa -adce -bdce -aggressive-instcombine -simplifycfg -loop-deletion -simplifycfg -correlated-propagation -adce -inline -instsimplify -instcombine -correlated-propagation #-autodiff_optimize=true

%-lowerprint.ll: %-unopt.ll
	../build/bin/opt $^ -indvars -load=../enzyme/build/Enzyme/LLVMEnzyme-7.so -enzyme -verify -mem2reg -sroa -adce -bdce -aggressive-instcombine -S -o $@ -instsimplify -correlated-propagation -early-cse-memssa -adce -bdce -aggressive-instcombine -simplifycfg -loop-deletion -simplifycfg -correlated-propagation -deadargelim -adce -bdce -aggressive-instcombine -inline -O3 -print-before-all -print-after-all

%-lowerq.ll: %-unopt.ll
	../build/bin/opt $^ -indvars -load=../enzyme/build/Enzyme/LLVMEnzyme-7.so -enzyme -verify -mem2reg -sroa -adce -bdce -aggressive-instcombine -S -instsimplify -correlated-propagation -early-cse-memssa -adce -bdce -aggressive-instcombine -simplifycfg -loop-deletion -simplifycfg -correlated-propagation -deadargelim -adce -bdce -aggressive-instcombine -inline -o moobar.ll
	../build/bin/opt moobar.ll -S -o $@ -O1 -print-before-all -print-after-all

%-lower.ll: %-unopt.ll
	../build/bin/opt $^ -indvars -load=../enzyme/build/Enzyme/LLVMEnzyme-7.so -enzyme -verify -mem2reg -sroa -adce -bdce -aggressive-instcombine -S -o - -enzyme_print=1 | ../build/bin/opt - -S -o $@ -instsimplify -correlated-propagation -early-cse-memssa -adce -bdce -aggressive-instcombine -simplifycfg -loop-deletion -simplifycfg -correlated-propagation -deadargelim -adce -bdce -aggressive-instcombine -inline -O3
	#../build/bin/opt $^ -indvars -autodiff_inline=true -lower-autodiff -verify -mem2reg -sroa -adce -bdce -aggressive-instcombine -S -o - | ../build/bin/opt - -S -o $@ -instsimplify -correlated-propagation -early-cse-memssa -adce -bdce -aggressive-instcombine -simplifycfg -loop-deletion -simplifycfg -correlated-propagation -deadargelim -adce -bdce -aggressive-instcombine -inline -O3
	#../build/bin/opt $^ -indvars -lower-autodiff -mem2reg -sroa -early-cse-memssa -adce -bdce -aggressive-instcombine -simplifycfg -correlated-propagation -deadargelim -adce -bdce -aggressive-instcombine -O3 -S -o $@

%.o: %-lower.ll
	../build/bin/clang $^ -o $@ -lm -lstdc++ -g -ggdb #-lblas #../../tapenade/ADFirstAidKit/*.o
	#../build/bin/clang $^ -o $@ -lm -ladept -lstdc++ -g -ggdb -lblas

clean:
	rm -f *.ll *.o

