-include ../tools.mk

ifeq ($(UNAME),Darwin)
PLUGIN_FLAGS := -C link-args=-Wl,-undefined,dynamic_lookup
endif

ifeq ($(findstring stage1,$(RUST_BUILD_STAGE)),stage1)
# ignore stage1
all:

else
# Windows doesn't correctly handle include statements with escaping paths,
# so this test will not get run on Windows.
ifdef IS_WINDOWS
all:
else
all: $(call NATIVE_STATICLIB,llvm-function-pass) $(call NATIVE_STATICLIB,llvm-module-pass)
	$(RUSTC) plugin.rs -C prefer-dynamic $(PLUGIN_FLAGS)
	$(RUSTC) main.rs

$(TMPDIR)/libllvm-function-pass.o:
	$(CXX) $(CFLAGS) $(LLVM_CXXFLAGS) -c llvm-function-pass.so.cc -o $(TMPDIR)/libllvm-function-pass.o

$(TMPDIR)/libllvm-module-pass.o:
	$(CXX) $(CFLAGS) $(LLVM_CXXFLAGS) -c llvm-module-pass.so.cc -o $(TMPDIR)/libllvm-module-pass.o
endif

endif
