-include ../tools.mk

# ignore windows due to libLLVM being present in PATH and the PATH and library path being the same
# (so fixing it is harder). See #57765 for context
ifndef IS_WINDOWS

# This test makes sure that we don't loose upstream object files when compiling
# staticlibs with -C linker-plugin-lto

all: staticlib.rs upstream.rs
	$(RUSTC) upstream.rs -C linker-plugin-lto -Ccodegen-units=1

	# Check No LTO
	$(RUSTC) staticlib.rs -C linker-plugin-lto -Ccodegen-units=1 -L. -o $(TMPDIR)/staticlib.a
	(cd $(TMPDIR); "$(LLVM_BIN_DIR)"/llvm-ar x ./staticlib.a)
	# Make sure the upstream object file was included
	ls $(TMPDIR)/upstream.*.rcgu.o

	# Cleanup
	rm $(TMPDIR)/*

	# Check ThinLTO
	$(RUSTC) upstream.rs -C linker-plugin-lto -Ccodegen-units=1 -Clto=thin
	$(RUSTC) staticlib.rs -C linker-plugin-lto -Ccodegen-units=1 -Clto=thin -L. -o $(TMPDIR)/staticlib.a
	(cd $(TMPDIR); "$(LLVM_BIN_DIR)"/llvm-ar x ./staticlib.a)
	ls $(TMPDIR)/upstream.*.rcgu.o

else

all:

endif
