# Test using -Cdlltool to change where raw-dylib looks for the dlltool binary.

# only-windows
# only-gnu
# needs-dlltool

include ../../run-make-fulldeps/tools.mk

all:
	$(RUSTC) --crate-type lib --crate-name raw_dylib_test lib.rs -Cdlltool=$(CURDIR)/script.cmd
	$(DIFF) output.txt "$(TMPDIR)"/output.txt
