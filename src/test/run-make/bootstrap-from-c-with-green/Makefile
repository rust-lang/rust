-include ../tools.mk

HOST_LIB_DIR=$(TMPDIR)/../../../stage$(RUST_BUILD_STAGE)/lib
# This overrides the LD_LIBRARY_PATH for RUN
TARGET_RPATH_DIR:=$(TARGET_RPATH_DIR):$(TMPDIR)

all:
	$(RUSTC) lib.rs
	ln -nsf $(call DYLIB,boot-*) $(call DYLIB,boot)
	$(CC) main.c -o $(call RUN_BINFILE,main) $(call RPATH_LINK_SEARCH,$(HOST_LIB_DIR)) -lboot
	$(call RUN,main)
	$(call REMOVE_DYLIBS,boot)
	$(call FAIL,main)
