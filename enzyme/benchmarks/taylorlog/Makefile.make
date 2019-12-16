# RUN: cd %desired_wd/taylorlog && LOAD="%loadEnzyme" make -B taylorlog-opt.ll taylorlog-results.txt VERBOSE=1 -f %s

.PHONY: time-* clean

clean:
	rm -f *.ll *.o
	
%-unopt.ll: %.cpp
	#clang++ -I../adept ../tapenade $^ -O2 -fno-unroll-loops -fno-exceptions -fno-vectorize -o $@
	clang++ -I../adept -I../tapenade $^ -O3 -fno-unroll-loops -fno-vectorize -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) -enzyme -o $@ -S
	
%-opt.ll: %-raw.ll
	opt $^ -O3 -o $@ -S

%.o: %-opt.ll
	clang++ $^ -o $@ -lblas ../tapenade/*.o

%-results.txt: %.o
	time ./$^ | tee $@
