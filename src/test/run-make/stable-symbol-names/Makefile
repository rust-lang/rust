-include ../tools.mk

# This test case makes sure that monomorphizations of the same function with the
# same set of generic arguments will have the same symbol names when
# instantiated in different crates.

dump-symbols = nm "$(TMPDIR)/lib$(1).rlib" \
             |  grep "some_test_function" \
             | sed "s/^[0-9a-f]\{8,16\}/00000000/" \
             | sort \
             > "$(TMPDIR)/$(1).nm"

all:
	$(RUSTC) stable-symbol-names1.rs
	$(RUSTC) stable-symbol-names2.rs
	$(call dump-symbols,stable_symbol_names1)
	$(call dump-symbols,stable_symbol_names2)
	cmp "$(TMPDIR)/stable_symbol_names1.nm" "$(TMPDIR)/stable_symbol_names2.nm"
