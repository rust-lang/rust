-include ../tools.mk

# Staticlibs don't include Rust object files from upstream crates if the same
# code was already pulled into the lib via LTO. However, the bug described in
# https://github.com/rust-lang/rust/issues/64153 lead to this exclusion not
# working properly if the upstream crate was compiled with an explicit filename
# (via `-o`).
#
# This test makes sure that functions defined in the upstream crates do not
# appear twice in the final staticlib when listing all the symbols from it.

all:
	$(RUSTC) --crate-type rlib upstream.rs -o $(TMPDIR)/libupstream.rlib -Ccodegen-units=1
	$(RUSTC) --crate-type staticlib downstream.rs -Clto -Ccodegen-units=1 -o $(TMPDIR)/libdownstream.a
	# Dump all the symbols from the staticlib into `syms`
	nm $(TMPDIR)/libdownstream.a > $(TMPDIR)/syms
	# Count the global instances of `issue64153_test_function`. There'll be 2
	# if the `upstream` object file got erronously included twice.
	grep -c -e "[[:space:]]T[[:space:]]issue64153_test_function" $(TMPDIR)/syms > $(TMPDIR)/count
	[ "$$(cat $(TMPDIR)/count)" -eq "1" ]
