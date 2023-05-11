# needs-matching-clang

# This test makes sure that cross-language inlining actually works by checking
# the generated machine code.

include ../tools.mk

all: cpp-executable rust-executable

cpp-executable:
	$(RUSTC) -Clinker-plugin-lto=on -o $(TMPDIR)/librustlib-xlto.a -Copt-level=2 -Ccodegen-units=1 ./rustlib.rs
	$(CLANG) -flto=thin -fuse-ld=lld -L $(TMPDIR) -lrustlib-xlto -o $(TMPDIR)/cmain ./cmain.c -O3
	# Make sure we don't find a call instruction to the function we expect to
	# always be inlined.
	"$(LLVM_BIN_DIR)"/llvm-objdump -d $(TMPDIR)/cmain | $(CGREP) -v -e "call.*rust_always_inlined"
	# As a sanity check, make sure we do find a call instruction to a
	# non-inlined function
	"$(LLVM_BIN_DIR)"/llvm-objdump -d $(TMPDIR)/cmain | $(CGREP) -e "call.*rust_never_inlined"

rust-executable:
	$(CLANG) ./clib.c -flto=thin -c -o $(TMPDIR)/clib.o -O2
	(cd $(TMPDIR); $(AR) crus ./libxyz.a ./clib.o)
	$(RUSTC) -Clinker-plugin-lto=on -L$(TMPDIR) -Copt-level=2 -Clinker=$(CLANG) -Clink-arg=-fuse-ld=lld ./main.rs -o $(TMPDIR)/rsmain
	"$(LLVM_BIN_DIR)"/llvm-objdump -d $(TMPDIR)/rsmain | $(CGREP) -e "call.*c_never_inlined"
	"$(LLVM_BIN_DIR)"/llvm-objdump -d $(TMPDIR)/rsmain | $(CGREP) -v -e "call.*c_always_inlined"
