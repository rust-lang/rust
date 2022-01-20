-include ../tools.mk

# aaa is a native static library
# bbb is a rlib
# ccc is a dylib
# ddd is an executable

all: $(call NATIVE_STATICLIB,aaa)
	$(RUSTC) bbb.rs --crate-type=rlib

	# Check that bbb does NOT contain the definition of `native_func`
	nm $(TMPDIR)/libbbb.rlib | $(CGREP) -ve "T _*native_func"
	nm $(TMPDIR)/libbbb.rlib | $(CGREP) -e "U _*native_func"

	# Check that aaa gets linked (either as `-l aaa` or `aaa.lib`) when building ccc.
	$(RUSTC) ccc.rs -C prefer-dynamic --crate-type=dylib --print link-args | $(CGREP) -e '-l[" ]*aaa|aaa\.lib'

	# Check that aaa does NOT get linked when building ddd.
	$(RUSTC) ddd.rs --print link-args | $(CGREP) -ve '-l[" ]*aaa|aaa\.lib'

	$(call RUN,ddd)
