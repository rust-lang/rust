# RUN: cd %desired_wd/hand && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" make -B hand-raw.ll -f %s
# TODO run

.PHONY: clean

clean:
	rm -f *.ll *.o results.txt

%-unopt.ll: %.c
	clang $(BENCH) $^ -ffast-math -O2 -fno-unroll-loops -fno-vectorize -o $@ -S -emit-llvm
#%-unopt.ll: %.cpp
	#clang++ $(BENCH) $^ -ffast-math -O2 -fno-unroll-loops -fno-vectorize -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) -enzyme -o $@ -S
	
%-opt.ll: %-raw.ll
	opt $^ -O2 -o $@ -S
	
%.o: %-opt.ll
	clang $^ -o $@

hand.o: hand-opt.ll
	clang++ $^ -o $@ -lblas $(BENCHLINK)

results.txt: hand.o
	./$^ | tee $@
