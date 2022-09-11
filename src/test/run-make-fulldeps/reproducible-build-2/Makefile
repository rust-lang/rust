include ../tools.mk

# ignore-musl
# ignore-windows
# Objects are reproducible but their path is not.

all:  \
	fat_lto \
	sysroot

fat_lto:
	rm -rf $(TMPDIR) && mkdir $(TMPDIR)
	$(RUSTC) reproducible-build-aux.rs
	$(RUSTC) reproducible-build.rs -C lto=fat
	cp $(TMPDIR)/reproducible-build $(TMPDIR)/reproducible-build-a
	$(RUSTC) reproducible-build.rs -C lto=fat
	cmp "$(TMPDIR)/reproducible-build-a" "$(TMPDIR)/reproducible-build" || exit 1

sysroot:
	rm -rf $(TMPDIR) && mkdir $(TMPDIR)
	$(RUSTC) reproducible-build-aux.rs
	$(RUSTC) reproducible-build.rs --crate-type rlib --sysroot $(shell $(RUSTC) --print sysroot) --remap-path-prefix=$(shell $(RUSTC) --print sysroot)=/sysroot
	cp -R $(shell $(RUSTC) --print sysroot) $(TMPDIR)/sysroot
	cp $(TMPDIR)/libreproducible_build.rlib $(TMPDIR)/libfoo.rlib
	$(RUSTC) reproducible-build.rs --crate-type rlib --sysroot $(TMPDIR)/sysroot --remap-path-prefix=$(TMPDIR)/sysroot=/sysroot
	cmp "$(TMPDIR)/libreproducible_build.rlib" "$(TMPDIR)/libfoo.rlib" || exit 1
