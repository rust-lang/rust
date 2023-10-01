# ignore-cross-compile
include ../tools.mk

# only-linux
# min-llvm-version: 16.0
#
# This tests debuginfo-compression.

all: zlib zstandard

zlib:
	test "`$(RUSTC) --crate-name=foo --crate-type=lib --emit=obj -C debuginfo=full -Z debuginfo-compression=zlib foo.rs 2>&1 | sed 's/.*unknown.*zlib.*/missing/' | head -n 1`" = missing || readelf -t $(TMPDIR)/foo.o | grep -q ZLIB

zstandard:
	test "`$(RUSTC) --crate-name=foo --crate-type=lib --emit=obj -C debuginfo=full -Z debuginfo-compression=zstd foo.rs 2>&1 | sed 's/.*unknown.*zstd.*/missing/' | head -n 1`" = missing || readelf -t $(TMPDIR)/foo.o | grep -q ZST
