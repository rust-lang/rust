-include ../tools.mk

# ignore windows: `ln` is actually `cp` on msys.
ifndef IS_WINDOWS

all:
	$(RUSTC) foo.rs
	mkdir -p $(TMPDIR)/other
	ln -nsf $(TMPDIR)/$(call DYLIB_GLOB,foo) $(TMPDIR)/other
	$(RUSTC) bar.rs -L $(TMPDIR)/other

else
all:

endif
