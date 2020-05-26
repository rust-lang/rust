# RUN: cd %desired_wd/lstm && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" make -B lstm-raw.ll results.txt -f %s

.PHONY: clean

clean:
	rm -f *.ll *.o results.txt

%-unopt.ll: %.cpp
	clang++ $(BENCH) $^ -O2 -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) -indvars -enzyme -mem2reg -early-cse -instcombine -adce -simplifycfg -loop-deletion -simplifycfg -o $@ -S

%-opt.ll: %-raw.ll
	opt $^ -O2 -early-cse-memssa -instcombine -indvars -o $@ -S

lstm.o: lstm-opt.ll
	clang++ $^ -o $@ -lblas $(BENCHLINK)

results.txt: lstm.o
	./$^ | tee $@
