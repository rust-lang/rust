# ignore-cross-compile
include ../tools.mk

# only-linux
# min-llvm-version: 16.0
#
# This tests debuginfo-compression.

all: zlib zstandard

zlib:
	$(RUSTC) --crate-name=foo --crate-type=lib --emit=obj -C debuginfo=full -Z debuginfo-compression=zlib foo.rs
	test "`ld.lld --compress-debug-sections=zlib 2>&1 | sed 's/.*unknown.*zlib/missing/'`" = missing || readelf -t $(TMPDIR)/foo.o | grep -q ZLIB

zstandard:
	$(RUSTC) --crate-name=foo --crate-type=lib --emit=obj -C debuginfo=full -Z debuginfo-compression=zstd foo.rs
	test "`ld.lld --compress-debug-sections=zstd 2>&1 | sed 's/.*unknown.*zstd/missing/'`" = missing || readelf -t $(TMPDIR)/foo.o | grep -q ZST
