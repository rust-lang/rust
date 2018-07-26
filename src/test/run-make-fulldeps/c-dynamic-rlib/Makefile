-include ../tools.mk

# This overrides the LD_LIBRARY_PATH for RUN
TARGET_RPATH_DIR:=$(TARGET_RPATH_DIR):$(TMPDIR)

# This hits an assertion in the linker on older versions of osx apparently
ifeq ($(shell uname),Darwin)
all:
	echo ignored
else
all: $(call DYLIB,cfoo)
	$(RUSTC) foo.rs
	$(RUSTC) bar.rs
	$(call RUN,bar)
	$(call REMOVE_DYLIBS,cfoo)
	$(call FAIL,bar)
endif
