# RUN: cd %desired_wd/taylorlog && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" make -B results.txt VERBOSE=1 -f %s

.PHONY: clean

clean:
	rm -f *.ll *.o results.txt
	
%-unopt.ll: %.cpp
	clang++ $(BENCH) $^ -ffast-math -O2 -fno-unroll-loops -fno-vectorize -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) -enzyme -o $@ -S
	
%-opt.ll: %-raw.ll
	opt $^ -O2 -o $@ -S
	
taylorlog.o: taylorlog-opt.ll
	clang++ $^ -o $@ -lblas $(BENCHLINK)

results.txt: taylorlog.o
	./$^ 10000000 | tee $@
