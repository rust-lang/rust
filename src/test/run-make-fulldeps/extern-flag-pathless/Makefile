include ../tools.mk

# Test mixing pathless --extern with paths.

all:
	$(RUSTC) bar-static.rs --crate-name=bar --crate-type=rlib
	$(RUSTC) bar-dynamic.rs --crate-name=bar --crate-type=dylib -C prefer-dynamic
	# rlib preferred over dylib
	$(RUSTC) foo.rs --extern bar
	$(call RUN,foo) | $(CGREP) 'static'
	$(RUSTC) foo.rs --extern bar=$(TMPDIR)/libbar.rlib --extern bar
	$(call RUN,foo) | $(CGREP) 'static'
	# explicit --extern overrides pathless
	$(RUSTC) foo.rs --extern bar=$(call DYLIB,bar) --extern bar
	$(call RUN,foo) | $(CGREP) 'dynamic'
	# prefer-dynamic does what it says
	$(RUSTC) foo.rs --extern bar -C prefer-dynamic
	$(call RUN,foo) | $(CGREP) 'dynamic'
