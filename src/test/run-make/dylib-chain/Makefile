-include ../tools.mk

all:
	$(RUSTC) m1.rs
	$(RUSTC) m2.rs
	$(RUSTC) m3.rs
	$(RUSTC) m4.rs
	$(call RUN,m4)
	$(call REMOVE_DYLIBS,m1)
	$(call REMOVE_DYLIBS,m2)
	$(call REMOVE_DYLIBS,m3)
	$(call FAIL,m4)
