-include ../../run-make-fulldeps/tools.mk

# How to manually run this
# $ ./x.py test src/test/run-make/rustdoc-test-builder

all:
	# Rustc isn't available in UI tests because the only compiler available is the
	# one built from source, which isn't in PATH.  This needs to be a run-make
	# test so rustc is available.
	$(RUSTDOC) --test -Z unstable-options --test-builder $(RUSTC_ORIGINAL) doctest.rs
