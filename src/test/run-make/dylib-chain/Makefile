-include ../tools.mk

all:
	$(RUSTC) m1.rs
	$(RUSTC) m2.rs
	$(RUSTC) m3.rs
	$(RUSTC) m4.rs
	$(call RUN,m4)
	rm $(TMPDIR)/$(call DYLIB_GLOB,m1)
	rm $(TMPDIR)/$(call DYLIB_GLOB,m2)
	rm $(TMPDIR)/$(call DYLIB_GLOB,m3)
	$(call FAIL,m4)
