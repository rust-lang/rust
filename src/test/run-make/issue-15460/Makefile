-include ../tools.mk

all: $(TMPDIR)/libfoo.a
	$(RUSTC) foo.rs -C extra-filename=-383hf8
	$(RUSTC) bar.rs
	$(call RUN,bar)
