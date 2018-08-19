-include ../tools.mk

all: $(TMPDIR)/libbar.a
	$(RUSTC) foo.rs -lstatic=bar
	$(RUSTC) main.rs
	$(call RUN,main)
