-include ../tools.mk

all: $(call NATIVE_STATICLIB,llvm-pass)
	$(RUSTC) plugin.rs -C prefer-dynamic
	$(RUSTC) main.rs

$(TMPDIR)/libllvm-pass.o:
	$(CXX) $(LLVM_CXXFLAGS) -c llvm-pass.so.cc -o $(TMPDIR)/libllvm-pass.o
