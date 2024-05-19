include ../tools.mk

# ignore-windows

export RUSTC := $(RUSTC_ORIGINAL)
export LD_LIBRARY_PATH := $(HOST_RPATH_DIR)
export TMPDIR := $(TMPDIR)

all:
	bash check.sh
