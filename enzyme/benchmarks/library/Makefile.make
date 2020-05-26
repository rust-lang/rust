# RUN: cd %desired_wd/library && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" make -B results.txt VERBOSE=1 -f %s

.PHONY: clean

clean:
	rm -f *.ll *.o results.txt *.o

%.o: %.c
	clang -c $(BENCH) $^ -ffast-math -O2 -fno-unroll-loops -fno-vectorize -o $@ -flto

merged.ll: mylib.o library.o
	clang mylib.o library.o -o $@ -S -emit-llvm

raw.ll: merged.ll
	opt $^ $(LOAD) -enzyme -mem2reg -early-cse -correlated-propagation -aggressive-instcombine -adce -loop-deletion -o $@ -S

opt.ll: raw.ll
	opt $^ -O2 -o $@ -S

library.exe: opt.ll
	clang $^ -o $@ -lblas $(BENCHLINK)

results.txt: library.exe
	./$^ 10000000 | tee $@
