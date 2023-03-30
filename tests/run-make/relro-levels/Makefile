include ../tools.mk

# only-linux
#
# This tests the different -Zrelro-level values, and makes sure that they work properly.

all:
	# Ensure that binaries built with the full relro level links them with both
	# RELRO and BIND_NOW for doing eager symbol resolving.
	$(RUSTC) -Zrelro-level=full hello.rs
	readelf -l $(TMPDIR)/hello | grep -q GNU_RELRO
	readelf -d $(TMPDIR)/hello | grep -q BIND_NOW

	$(RUSTC) -Zrelro-level=partial hello.rs
	readelf -l $(TMPDIR)/hello | grep -q GNU_RELRO

	# Ensure that we're *not* built with RELRO when setting it to off.  We do
	# not want to check for BIND_NOW however, as the linker might have that
	# enabled by default.
	$(RUSTC) -Zrelro-level=off hello.rs
	! readelf -l $(TMPDIR)/hello | grep -q GNU_RELRO
