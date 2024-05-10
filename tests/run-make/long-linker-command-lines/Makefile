# ignore-cross-compile
include ../tools.mk

export LD_LIBRARY_PATH := $(HOST_RPATH_DIR)

all:
	$(RUSTC) foo.rs -g -O
	RUSTC="$(RUSTC_ORIGINAL)" $(call RUN,foo)
