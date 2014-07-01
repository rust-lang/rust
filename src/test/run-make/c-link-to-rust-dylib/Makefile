-include ../tools.mk

HOST_LIB_DIR=$(TMPDIR)/../../../stage$(RUST_BUILD_STAGE)/lib

all:
	$(RUSTC) foo.rs
	$(CC) bar.c -lfoo -o $(call RUN_BINFILE,bar) $(call RPATH_LINK_SEARCH,$(HOST_LIB_DIR)) -Wl,-rpath,$(TMPDIR)
	$(call RUN,bar)
	$(call REMOVE_DYLIBS,foo)
	$(call FAIL,bar)
