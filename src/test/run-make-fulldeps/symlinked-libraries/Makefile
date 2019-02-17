-include ../tools.mk

# ignore-windows
# `ln` is actually `cp` on msys.

all:
	$(RUSTC) foo.rs -C prefer-dynamic
	mkdir -p $(TMPDIR)/other
	ln -nsf $(TMPDIR)/$(call DYLIB_GLOB,foo) $(TMPDIR)/other
	$(RUSTC) bar.rs -L $(TMPDIR)/other
