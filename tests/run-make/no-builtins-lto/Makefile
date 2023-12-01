include ../tools.mk

# only-x86_64

# We want to check that `no_builtins` is correctly participating in LTO.
# First, verify that the `foo::foo` symbol can be found when linking.
# Next, verify that `memcpy` can be customized using `no_builtins` under LTO.
# Others will use the built-in memcpy.

all:
	$(RUSTC) -C linker-plugin-lto -C opt-level=2 -C debuginfo=0 foo.rs
	$(RUSTC) -C linker-plugin-lto -C opt-level=2 -C debuginfo=0 no_builtins.rs
	$(RUSTC) main.rs -C lto -C opt-level=2 -C debuginfo=0 -C save-temps -C metadata=1 -C codegen-units=1
	"$(LLVM_BIN_DIR)"/llvm-dis $(TMPDIR)/main.main.*-cgu.0.rcgu.lto.input.bc -o $(TMPDIR)/lto.ll
	cat "$(TMPDIR)"/lto.ll | "$(LLVM_FILECHECK)" filecheck.lto.txt
