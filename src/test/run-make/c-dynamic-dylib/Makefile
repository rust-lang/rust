-include ../tools.mk

# needed so that libfoo can find libcfoo
ifeq ($(shell uname),Linux)
export LD_LIBRARY_PATH := $(TMPDIR)
endif

# This hits an assertion in the linker on older versions of osx apparently
ifeq ($(shell uname),Darwin)
all:
	echo ignored
else
all: $(call DYLIB,cfoo)
	$(RUSTC) foo.rs
	$(RUSTC) bar.rs
	$(call RUN,bar)
	rm $(TMPDIR)/$(call DYLIB_GLOB,cfoo)
	$(call FAIL,bar)
endif
