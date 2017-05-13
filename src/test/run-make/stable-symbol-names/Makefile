-include ../tools.mk

# The following command will:
#  1. dump the symbols of a library using `nm`
#  2. extract only those lines that we are interested in via `grep`
#  3. from those lines, extract just the symbol name via `sed`
#     (symbol names always start with "_ZN" and end with "E")
#  4. sort those symbol names for deterministic comparison
#  5. write the result into a file

dump-symbols = nm "$(TMPDIR)/lib$(1).rlib" \
             | grep -E "some_test_function|Bar|bar" \
             | sed "s/.*\(_ZN.*E\).*/\1/" \
             | sort \
             > "$(TMPDIR)/$(1).nm"

all:
	$(RUSTC) stable-symbol-names1.rs
	$(RUSTC) stable-symbol-names2.rs
	$(call dump-symbols,stable_symbol_names1)
	$(call dump-symbols,stable_symbol_names2)
	cmp "$(TMPDIR)/stable_symbol_names1.nm" "$(TMPDIR)/stable_symbol_names2.nm"
