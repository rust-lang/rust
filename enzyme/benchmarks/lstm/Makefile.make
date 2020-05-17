# RUN: cd %desired_wd/lstm && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" make -B lstm-raw.ll -f %s
# TODO run

.PHONY: clean

clean:
	rm -f *.ll *.o results.txt

%-unopt.ll: %.c
	clang $(BENCH) $^ -ffast-math -O2 -fno-unroll-loops -fno-vectorize -o $@ -S -emit-llvm
#%-unopt.ll: %.cpp
	#clang++ $(BENCH) $^ -ffast-math -O2 -fno-unroll-loops -fno-vectorize -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) -enzyme -mem2reg -o $@ -S
	
%-opt.ll: %-raw.ll
	opt $^ -O2 -early-cse-memssa -instcombine -indvars -o $@ -S
	
%.o: %-opt.ll
	clang $^ -o $@

lstm.o: lstm-opt.ll
	clang++ $^ -o $@ -lblas $(BENCHLINK)

results.txt: lstm.o
	./$^ | tee $@
