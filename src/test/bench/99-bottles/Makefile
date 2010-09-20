RC:=../../../rustboot
RFLAGS:=-L ../../..
TARGETS:= 99bob-simple 99bob-iter 99bob-tail 99bob-pattern
TARGET_X86:=$(addsuffix .x86,$(TARGETS))
TARGET_LLVM:=$(addsuffix .llvm,$(TARGETS))

all : x86s llvms

clean:
	rm  $(TARGET_X86) $(TARGET_LLVM)

x86s : $(TARGET_X86) 

llvms:  $(TARGET_LLVM)

%.x86 : %.rs
	$(RC) $(RFLAGS) $^ -o $@

%.llvm : %.rs
	$(RC) $(RFLAGS) -llvm $^ -o $@