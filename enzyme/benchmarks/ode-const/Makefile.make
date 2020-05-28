# RUN: cd %desired_wd/ode-const && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" make -B ode-raw.ll ode-opt.ll results.txt VERBOSE=1 -f %s

.PHONY: clean

clean:
	rm -f *.ll *.o results.txt

ode-adept-unopt.ll: ode-adept.cpp
	clang++ $(BENCH) $^ -O2 -fno-use-cxa-atexit -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm
	#clang++ $(BENCH) $^ -O1 -Xclang -disable-llvm-passes -fno-use-cxa-atexit -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm

ode-unopt.ll: ode.cpp
	clang++ $(BENCH) $^ -O2 -fno-use-cxa-atexit -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm
	#clang++ $(BENCH) $^ -fno-exceptions -Xclang -disable-llvm-passes -fno-use-cxa-atexit -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm

ode-raw.ll: ode-adept-unopt.ll ode-unopt.ll
	opt ode-unopt.ll -mem2reg -sroa -deadargelim -o ode-pp.ll -S
	opt ode-pp.ll $(LOAD) -enzyme -o ode-enzyme.ll -S
	llvm-link ode-adept-unopt.ll ode-enzyme.ll -o $@

%-opt.ll: %-raw.ll
	opt-8 $^ -O2 -o $@ -S

ode.o: ode-opt.ll
	clang++ $^ -o $@ -lblas $(BENCHLINK)
	#clang++ -O2 $^ -o $@ -lblas $(BENCHLINK)

results.txt: ode.o
	./$^ 30000000 | tee $@
