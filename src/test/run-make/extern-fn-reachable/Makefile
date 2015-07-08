-include ../tools.mk

# This overrides the LD_LIBRARY_PATH for RUN
TARGET_RPATH_DIR:=$(TARGET_RPATH_DIR):$(TMPDIR)

all:
	$(RUSTC) dylib.rs -o $(TMPDIR)/libdylib.so -C prefer-dynamic
	$(RUSTC) main.rs -C prefer-dynamic
	$(call RUN,main)
