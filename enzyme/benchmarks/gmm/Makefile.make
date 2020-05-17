# RUN: cd %desired_wd/gmm && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" make -B gmm-raw.ll -f %s
# TODO run
#  - note haven't set up actual result gathering code yet

.PHONY: clean

clean:
	rm -f *.ll *.o results.txt
	
%-unopt.ll: %.c
	clang $(BENCH) $^ -ffast-math -O2 -fno-unroll-loops -fno-vectorize -o $@ -S -emit-llvm
#%-unopt.ll: %.cpp
	#clang++ $(BENCH) $^ -ffast-math -O2 -fno-unroll-loops -fno-vectorize -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) -indvars -enzyme -o $@ -S
	
%-opt.ll: %-raw.ll
	opt $^ -O2 -indvars -o $@ -S
	
%.o: %-c-opt.ll
	clang $^ -o $@

gmm.o: gmm-opt.ll
	clang++ $^ -o $@ -lblas $(BENCHLINK)

results.txt: gmm.o
	./$^ | tee $@
