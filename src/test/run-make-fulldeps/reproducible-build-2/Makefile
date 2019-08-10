-include ../tools.mk

# ignore-musl
# ignore-windows
# Objects are reproducible but their path is not.

all:  \
	fat_lto

fat_lto:
	rm -rf $(TMPDIR) && mkdir $(TMPDIR)
	$(RUSTC) reproducible-build-aux.rs
	$(RUSTC) reproducible-build.rs -C lto=fat
	cp $(TMPDIR)/reproducible-build $(TMPDIR)/reproducible-build-a
	$(RUSTC) reproducible-build.rs -C lto=fat
	cmp "$(TMPDIR)/reproducible-build-a" "$(TMPDIR)/reproducible-build" || exit 1
