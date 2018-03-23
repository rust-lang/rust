-include ../tools.mk

# The rust crate foo will link to the native library foo, while the rust crate
# bar will link to the native library bar. There is also a dependency between
# the native library bar to the natibe library foo.
#
# This test ensures that the ordering of -lfoo and -lbar on the command line is
# correct to complete the linkage. If passed as "-lfoo -lbar", then the 'foo'
# library will be stripped out, and the linkage will fail.

all: $(call NATIVE_STATICLIB,foo) $(call NATIVE_STATICLIB,bar)
	$(RUSTC) foo.rs
	$(RUSTC) bar.rs
	$(RUSTC) main.rs -Z print-link-args
