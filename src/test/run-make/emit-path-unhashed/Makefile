-include ../../run-make-fulldeps/tools.mk

OUT=$(TMPDIR)/emit

# --emit KIND=PATH should not affect crate hash vs --emit KIND
all: $(OUT)/a/libfoo.rlib $(OUT)/b/libfoo.rlib $(TMPDIR)/libfoo.rlib
	$(RUSTC) -Zls $(TMPDIR)/libfoo.rlib > $(TMPDIR)/base.txt
	$(RUSTC) -Zls $(OUT)/a/libfoo.rlib > $(TMPDIR)/a.txt
	$(RUSTC) -Zls $(OUT)/b/libfoo.rlib > $(TMPDIR)/b.txt

	diff $(TMPDIR)/base.txt $(TMPDIR)/a.txt
	diff $(TMPDIR)/base.txt $(TMPDIR)/b.txt

# Default output name
$(TMPDIR)/libfoo.rlib: foo.rs
	$(RUSTC) --emit link foo.rs

# Output named with -o
$(OUT)/a/libfoo.rlib: foo.rs
	mkdir -p $(OUT)/a
	$(RUSTC) --emit link -o $@ foo.rs

# Output named with KIND=PATH
$(OUT)/b/libfoo.rlib: foo.rs
	mkdir -p $(OUT)/b
	$(RUSTC) --emit link=$@ foo.rs
