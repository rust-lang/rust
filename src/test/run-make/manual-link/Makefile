-include ../tools.mk

all: $(TMPDIR)/libbar.a
	$(RUSTC) foo.rs -lbar:static
	$(RUSTC) main.rs
	$(call RUN,main)

