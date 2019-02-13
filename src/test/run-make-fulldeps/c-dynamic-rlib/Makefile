-include ../tools.mk

# ignore-macos
#
# This hits an assertion in the linker on older versions of osx apparently

# This overrides the LD_LIBRARY_PATH for RUN
TARGET_RPATH_DIR:=$(TARGET_RPATH_DIR):$(TMPDIR)

all: $(call DYLIB,cfoo)
	$(RUSTC) foo.rs
	$(RUSTC) bar.rs
	$(call RUN,bar)
	$(call REMOVE_DYLIBS,cfoo)
	$(call FAIL,bar)
