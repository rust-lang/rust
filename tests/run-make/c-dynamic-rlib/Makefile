# This test checks that dynamic Rust linking with C does not encounter any errors, with static dependencies given preference over dynamic. (This is the default behaviour.)
# See https://github.com/rust-lang/rust/issues/10434

# ignore-cross-compile
include ../tools.mk

# ignore-apple
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
