# ignore-cross-compile
include ../tools.mk

# Test mixing pathless --extern with paths.

# Test for static linking by checking that the binary runs if the dylib
# is removed and test for dynamic linking by checking that the binary
# fails to run if the dylib is removed.

all:
	$(RUSTC) bar.rs --crate-type=rlib --crate-type=dylib -Cprefer-dynamic

	# rlib preferred over dylib
	$(RUSTC) foo.rs --extern bar
	mv $(call DYLIB,bar) $(TMPDIR)/bar.tmp
	$(call RUN,foo)
	mv $(TMPDIR)/bar.tmp $(call DYLIB,bar)

	$(RUSTC) foo.rs --extern bar=$(TMPDIR)/libbar.rlib --extern bar
	mv $(call DYLIB,bar) $(TMPDIR)/bar.tmp
	$(call RUN,foo)
	mv $(TMPDIR)/bar.tmp $(call DYLIB,bar)

	# explicit --extern overrides pathless
	$(RUSTC) foo.rs --extern bar=$(call DYLIB,bar) --extern bar
	mv $(call DYLIB,bar) $(TMPDIR)/bar.tmp
	$(call FAIL,foo)
	mv $(TMPDIR)/bar.tmp $(call DYLIB,bar)

	# prefer-dynamic does what it says
	$(RUSTC) foo.rs --extern bar -C prefer-dynamic
	mv $(call DYLIB,bar) $(TMPDIR)/bar.tmp
	$(call FAIL,foo)
	mv $(TMPDIR)/bar.tmp $(call DYLIB,bar)
