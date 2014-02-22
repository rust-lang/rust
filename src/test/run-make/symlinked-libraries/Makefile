-include ../tools.mk

all:
	$(RUSTC) foo.rs
	mkdir -p $(TMPDIR)/other
	ln -nsf $(TMPDIR)/$(call DYLIB_GLOB,foo) $(TMPDIR)/other
	$(RUSTC) bar.rs -L $(TMPDIR)/other
