# RUN: cd %desired_wd/library && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" make -B results.txt VERBOSE=1 -f %s

.PHONY: clean

clean:
	rm -f *.ll *.o results.txt *.o

%.o: %.c
	clang -flto -c $(BENCH) $^ -ffast-math -O2 -fno-unroll-loops -fno-vectorize -o $@

# in fto mode these are just bc files by another name
combined.bc: library.o mylib.o
	llvm-link $^ -o $@

raw.ll: combined.bc
	opt $^ $(LOAD) -enzyme -mem2reg -early-cse -correlated-propagation -aggressive-instcombine -adce -loop-deletion -o $@ -S

opt.ll: raw.ll
	opt $^ -O2 -o $@ -S

library.exe: opt.ll
	clang $^ -o $@ -lblas $(BENCHLINK)

results.txt: library.exe
	./$^ 10000000 | tee $@
