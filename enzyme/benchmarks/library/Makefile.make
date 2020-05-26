# RUN: cd %desired_wd/library && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" make -B results.txt VERBOSE=1 -f %s

.PHONY: clean

clean:
	rm -f *.ll *.o results.txt *.o

mylib.so: mylib.c
	clang -shared $^ -ffast-math -O2 -fno-unroll-loops -fno-vectorize -fembed-bitcode -o $@

%.o: %.c
	clang -c $(BENCH) $^ -ffast-math -O2 -fno-unroll-loops -fno-vectorize -o $@ -flto

library.ll: library.o
	clang library.o -o $@ -S -emit-llvm

raw.ll: library.ll
	opt $^ $(LOAD) -enzyme -mem2reg -early-cse -correlated-propagation -aggressive-instcombine -adce -loop-deletion -o $@ -S

opt.ll: raw.ll
	opt $^ -O2 -o $@ -S

library.exe: opt.ll
	clang $^ -o $@ -lblas -L. -lmylib.so $(BENCHLINK)

results.txt: library.exe
	./$^ 10000000 | tee $@
