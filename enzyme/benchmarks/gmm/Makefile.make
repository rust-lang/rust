# RUN: cd %desired_wd/gmm && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" make -B gmm-unopt.ll gmm-raw.ll results.txt -f %s

.PHONY: clean

clean:
	rm -f *.ll *.o results.txt

%-unopt.ll: %.cpp
	clang++ $(BENCH) $^ -O2 -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm
	#clang++ $(BENCH) $^ -O1 -Xclang -disable-llvm-passes -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) -enzyme -o $@ -S

%-opt.ll: %-raw.ll
	opt $^ -o $@ -S
	#opt $^ -O2 -o $@ -S

gmm.o: gmm-opt.ll
	clang++ -O2 $^ -o $@ -lblas $(BENCHLINK)

results.txt: gmm.o
	./$^ | tee $@
