-include ../tools.mk

# Windows doesn't correctly handle include statements with escaping paths,
# so this test will not get run on Windows.
ifdef IS_WINDOWS
all:
else
all: $(call NATIVE_STATICLIB,llvm-pass)
	$(RUSTC) plugin.rs -C prefer-dynamic
	$(RUSTC) main.rs

$(TMPDIR)/libllvm-pass.o:
	$(CXX) $(CFLAGS) $(LLVM_CXXFLAGS) -c llvm-pass.so.cc -o $(TMPDIR)/libllvm-pass.o
endif
