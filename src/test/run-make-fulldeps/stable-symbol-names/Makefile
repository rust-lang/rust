-include ../tools.mk

# The following command will:
#  1. dump the symbols of a library using `nm`
#  2. extract only those lines that we are interested in via `grep`
#  3. from those lines, extract just the symbol name via `sed`
#     (symbol names always start with "_ZN" and end with "E")
#  4. sort those symbol names for deterministic comparison
#  5. write the result into a file

dump-symbols = nm "$(TMPDIR)/lib$(1).rlib" \
             | grep -E "$(2)" \
             | sed "s/.*\(_ZN.*E\).*/\1/" \
             | sort \
             > "$(TMPDIR)/$(1)$(3).nm"

# This test
# - compiles each of the two crates 2 times and makes sure each time we get
#   exactly the same symbol names
# - makes sure that both crates agree on the same symbol names for monomorphic
#   functions

all:
	$(RUSTC) stable-symbol-names1.rs
	$(call dump-symbols,stable_symbol_names1,generic_|mono_,_v1)
	rm $(TMPDIR)/libstable_symbol_names1.rlib
	$(RUSTC) stable-symbol-names1.rs
	$(call dump-symbols,stable_symbol_names1,generic_|mono_,_v2)
	cmp "$(TMPDIR)/stable_symbol_names1_v1.nm" "$(TMPDIR)/stable_symbol_names1_v2.nm"

	$(RUSTC) stable-symbol-names2.rs
	$(call dump-symbols,stable_symbol_names2,generic_|mono_,_v1)
	rm $(TMPDIR)/libstable_symbol_names2.rlib
	$(RUSTC) stable-symbol-names2.rs
	$(call dump-symbols,stable_symbol_names2,generic_|mono_,_v2)
	cmp "$(TMPDIR)/stable_symbol_names2_v1.nm" "$(TMPDIR)/stable_symbol_names2_v2.nm"

	$(call dump-symbols,stable_symbol_names1,mono_,_cross)
	$(call dump-symbols,stable_symbol_names2,mono_,_cross)
	cmp "$(TMPDIR)/stable_symbol_names1_cross.nm" "$(TMPDIR)/stable_symbol_names2_cross.nm"
