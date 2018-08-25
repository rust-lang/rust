-include ../tools.mk

ifeq ($(findstring stage1,$(RUST_BUILD_STAGE)),stage1)
# ignore stage1
all:

else
all:
	mkdir $(TMPDIR)/incremental-dir
	$(RUSTC) macro_def.rs
	$(RUSTC) reexport.rs
	$(RUSTC) main.rs -C incremental=$(TMPDIR)/incremental-dir
endif
