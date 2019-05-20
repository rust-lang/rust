.SUFFIXES:

.PHONY : clean

%-unopt.ll: %.c
	../build/bin/clang -fno-unroll-loops $^ -O3 -fno-vectorize -fno-slp-vectorize -S -emit-llvm -o $@ -ffast-math

%-unopt.ll: %.cpp
	../build/bin/clang++ -fno-exceptions -fno-unroll-loops $^ -O3 -fno-vectorize -fno-slp-vectorize -S -emit-llvm -o $@ -ffast-math

%-lower.ll: %-unopt.ll
	../build/bin/opt $^ -indvars -lower-autodiff -mem2reg -sroa -early-cse-memssa -adce -bdce -aggressive-instcombine -simplifycfg -correlated-propagation -deadargelim -adce -bdce -aggressive-instcombine -O3 -S -o $@

%.o: %-lower.ll
	../build/bin/clang $^ -o $@ -lm -ladept -lstdc++

clean:
	rm -f *.ll *.o

