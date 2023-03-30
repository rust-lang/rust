include ../tools.mk

# ignore-macos
#
# This hits an assertion in the linker on older versions of osx apparently

all: $(call DYLIB,cfoo)
	$(RUSTC) foo.rs -C prefer-dynamic
	$(RUSTC) bar.rs
	$(call RUN,bar)
	$(call REMOVE_DYLIBS,cfoo)
	$(call FAIL,bar)
