# RUN: cd %desired_wd/nn && pwd && ls && LOAD="%loadEnzyme" make nn-results.txt VERBOSE=1 -f %s

.PHONY: time-* clean

clean:
	rm -f *.ll *.o
	
%-c-unopt.ll: %.c
	clang $^ -O2 -fno-unroll-loops -fno-vectorize -o $@ -S -emit-llvm

%-cpp-unopt.ll: %.cpp
	#clang++ -I../adept ../tapenade $^ -O2 -fno-unroll-loops -fno-exceptions -fno-vectorize -o $@
	clang++ -I../adept -I../tapenade $^ -O2 -fno-unroll-loops -fno-vectorize -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) -enzyme -o $@ -S
	
%-opt.ll: %-raw.ll
	opt $^ -O2 -o $@ -S
	
%.o: %-c-opt.ll
	clang $^ -o $@

%.o: %-cpp-opt.ll
	clang++ $^ -o $@ -lblas ../tapenade/*.o

%-results.txt: %.o
	time ./$^ | tee $@
