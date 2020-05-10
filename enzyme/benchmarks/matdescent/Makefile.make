# RUN: cd %desired_wd/matdescent && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" make -B results.txt -f %s

.PHONY: clean

clean:
	rm -f *.ll *.o results.txt
	
%-unopt.ll: %.cpp
	clang++ $(BENCH) $^ -ffast-math -O2 -fno-unroll-loops -fno-vectorize -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) -enzyme -mem2reg -simplifycfg -early-cse -correlated-propagation -instcombine -adce -o $@ -S
	
%-opt.ll: %-raw.ll
	opt $^ -O2 -o $@ -S
	
matdescent.o: matdescent-opt.ll
	clang++ $^ -o $@ -lblas $(BENCHLINK)

results.txt: matdescent.o
	./$^ | tee $@
