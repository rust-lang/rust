# RUN: cd %desired_wd/gmm && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" make -B gmm-unopt.ll gmm-raw.ll results.txt -f %s
# TODO run
#  - note haven't set up actual result gathering code yet

.PHONY: clean

clean:
	rm -f *.ll *.o results.txt

%-unopt.ll: %.cpp
	clang++ $(BENCH) $^ -O2 -fno-exceptions -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) -indvars -enzyme -mem2reg -early-cse -instcombine -adce -simplifycfg -loop-deletion -simplifycfg -o $@ -S

%-opt.ll: %-raw.ll
	opt $^ -O2 -indvars -o $@ -S

gmm.o: gmm-opt.ll
	clang++ $^ -o $@ -lblas $(BENCHLINK)

results.txt: gmm.o
	./$^ | tee $@
