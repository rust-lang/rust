# only-x86_64-pc-windows-msvc

include ../tools.mk

# assert that import list for simple executable not changed silently
# FIXME: remove "File:" from diff by making path reproducible: now it's broken

all:
	$(RUSTC) empty.rs
	"$(LLVM_BIN_DIR)"/llvm-readobj --coff-imports $(TMPDIR)/empty.exe > "$(TMPDIR)/actual.empty.txt"
	diff -u -I "File:" -I "ImportLookupTableRVA:" -I "ImportAddressTableRVA:" expected.empty.txt "$(TMPDIR)/actual.empty.txt"
