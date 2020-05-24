# RUN: cd %desired_wd/ode-const && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" make -B ode-raw.ll ode-opt.ll results.txt VERBOSE=1 -f %s

.PHONY: clean

clean:
	rm -f *.ll *.o results.txt

%-unopt.ll: %.cpp
	clang++ $(BENCH) $^ -O2 -fno-use-cxa-atexit -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm
	#clang++ $(BENCH) $^ -O2 -fno-use-cxa-atexit -fno-exceptions -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) -enzyme -mem2reg -early-cse -correlated-propagation -aggressive-instcombine -adce -loop-deletion -o $@ -S

%-opt.ll: %-raw.ll
	#opt $^ -O2 -o $@ -S
	opt $^ -o $@ -S

ode.o: ode-opt.ll
	clang++ $^ -o $@ -lblas $(BENCHLINK)

results.txt: ode.o
	./$^ 30000000 | tee $@
