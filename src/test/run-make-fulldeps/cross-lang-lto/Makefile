
# min-llvm-version 4.0
# ignore-mingw

-include ../tools.mk

# This test makes sure that the expected .llvmbc sections for use by
# linker-based LTO are available in object files when compiling with
# -Z cross-lang-lto

LLVMBC_SECTION_NAME=\\.llvmbc

ifeq ($(UNAME),Darwin)
	LLVMBC_SECTION_NAME=__bitcode
endif


OBJDUMP=llvm-objdump
SECTION_HEADERS=$(OBJDUMP) -section-headers

BUILD_LIB=$(RUSTC) lib.rs -Copt-level=2 -Z cross-lang-lto=no-link -Ccodegen-units=1

BUILD_EXE=$(RUSTC) main.rs -Copt-level=2 -Z cross-lang-lto=no-link -Ccodegen-units=1 --emit=obj

all: staticlib staticlib-fat-lto staticlib-thin-lto rlib exe cdylib rdylib

staticlib: lib.rs
	$(BUILD_LIB) --crate-type=staticlib -o $(TMPDIR)/liblib.a
	[ "$$($(SECTION_HEADERS) $(TMPDIR)/liblib.a | grep -c $(LLVMBC_SECTION_NAME))" -ne "0" ]

staticlib-fat-lto: lib.rs
	$(BUILD_LIB) --crate-type=staticlib -o $(TMPDIR)/liblib-fat-lto.a -Clto=fat
	[ "$$($(SECTION_HEADERS) $(TMPDIR)/liblib-fat-lto.a | grep -c $(LLVMBC_SECTION_NAME))" -ne "0" ]

staticlib-thin-lto: lib.rs
	$(BUILD_LIB) --crate-type=staticlib -o $(TMPDIR)/liblib-thin-lto.a -Clto=thin
	[ "$$($(SECTION_HEADERS) $(TMPDIR)/liblib-thin-lto.a | grep -c $(LLVMBC_SECTION_NAME))" -ne "0" ]

rlib: lib.rs
	$(BUILD_LIB) --crate-type=rlib -o $(TMPDIR)/liblib.rlib
	[ "$$($(SECTION_HEADERS) $(TMPDIR)/liblib.rlib | grep -c $(LLVMBC_SECTION_NAME))" -ne "0" ]

cdylib: lib.rs
	$(BUILD_LIB) --crate-type=cdylib --emit=obj -o $(TMPDIR)/cdylib.o
	[ "$$($(SECTION_HEADERS) $(TMPDIR)/cdylib.o | grep -c $(LLVMBC_SECTION_NAME))" -ne "0" ]

rdylib: lib.rs
	$(BUILD_LIB) --crate-type=dylib --emit=obj -o $(TMPDIR)/rdylib.o
	[ "$$($(SECTION_HEADERS) $(TMPDIR)/rdylib.o | grep -c $(LLVMBC_SECTION_NAME))" -ne "0" ]

exe: lib.rs
	$(BUILD_EXE) -o $(TMPDIR)/exe.o
	[ "$$($(SECTION_HEADERS) $(TMPDIR)/exe.o | grep -c $(LLVMBC_SECTION_NAME))" -ne "0" ]
