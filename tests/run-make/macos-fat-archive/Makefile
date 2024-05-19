# only-macos

include ../tools.mk

"$(TMPDIR)"/libnative-library.a: native-library.c
	$(CC) -arch arm64 -arch x86_64 native-library.c -c -o "$(TMPDIR)"/native-library.o
	$(AR) crs "$(TMPDIR)"/libnative-library.a "$(TMPDIR)"/native-library.o

all: "$(TMPDIR)"/libnative-library.a
	$(RUSTC) lib.rs --crate-type=lib -L "native=$(TMPDIR)" -l static=native-library
