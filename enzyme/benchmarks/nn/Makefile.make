; RUN: LOAD="%loadEnzyme" make time-nn -f %s

.PHONY: time

%-c-unopt.ll: %.c
	clang $^ -O2 -fno-unroll-loops -fno-exceptions -fno-vectorize -o $@

%-cpp-unopt.ll: %.cpp
	clang++ $^ -O2 -fno-unroll-loops -fno-exceptions -fno-vectorize -o $@

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) -enzyme -o $@ 
	
%-opt.ll: %-raw.ll
	opt $^ -O2 -o $@ 
	
%.o: %-c-opt.ll
	clang $^ -O2 -o $@ 

%.o: %-c-opt.ll
	clang++ $^ -O2 -o $@ 

time-%: %.o
	time ./$^
