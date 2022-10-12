include ../tools.mk

all:
	$(RUSTC) m1.rs
	$(RUSTC) m2.rs
	$(RUSTC) m3.rs
	$(RUSTC) m4.rs
	$(call RUN,m4)
	rm $(TMPDIR)/*lib
	$(call RUN,m4)
