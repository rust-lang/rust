
-include ../tools.mk

# This test makes sure that the object files we generate are actually
# LLVM bitcode files (as used by linker LTO plugins) when compiling with
# -Z cross-lang-lto.

ASSERT_IS_BITCODE_OBJ=llvm-bcanalyzer # this only succeeds for bitcode files
EXTRACT_OBJS=(cd $(TMPDIR); rm -f ./*.o; llvm-ar x $(1))

BUILD_LIB=$(RUSTC) lib.rs -Copt-level=2 -Z cross-lang-lto=on -Ccodegen-units=1
BUILD_EXE=$(RUSTC) main.rs -Copt-level=2 -Z cross-lang-lto=on -Ccodegen-units=1 --emit=obj

all: staticlib staticlib-fat-lto staticlib-thin-lto rlib exe cdylib rdylib

staticlib: lib.rs
	$(BUILD_LIB) --crate-type=staticlib -o $(TMPDIR)/liblib.a
	$(call EXTRACT_OBJS, liblib.a)
	$(ASSERT_IS_BITCODE_OBJ) $(TMPDIR)/liblib.lib0.rcgu.o

staticlib-fat-lto: lib.rs
	$(BUILD_LIB) --crate-type=staticlib -o $(TMPDIR)/liblib-fat-lto.a -Clto=fat
	$(call EXTRACT_OBJS, liblib-fat-lto.a)
	$(ASSERT_IS_BITCODE_OBJ) $(TMPDIR)/liblib-fat-lto.lib0.rcgu.o

staticlib-thin-lto: lib.rs
	$(BUILD_LIB) --crate-type=staticlib -o $(TMPDIR)/liblib-thin-lto.a -Clto=thin
	$(call EXTRACT_OBJS, liblib-thin-lto.a)
	$(ASSERT_IS_BITCODE_OBJ) $(TMPDIR)/liblib-thin-lto.lib0.rcgu.o

rlib: lib.rs
	$(BUILD_LIB) --crate-type=rlib -o $(TMPDIR)/liblib.rlib
	$(call EXTRACT_OBJS, liblib.rlib)
	$(ASSERT_IS_BITCODE_OBJ) $(TMPDIR)/liblib.lib0.rcgu.o

cdylib: lib.rs
	$(BUILD_LIB) --crate-type=cdylib --emit=obj -o $(TMPDIR)/cdylib.o
	$(ASSERT_IS_BITCODE_OBJ) $(TMPDIR)/cdylib.o

rdylib: lib.rs
	$(BUILD_LIB) --crate-type=dylib --emit=obj -o $(TMPDIR)/rdylib.o
	$(ASSERT_IS_BITCODE_OBJ) $(TMPDIR)/rdylib.o

exe: lib.rs
	$(BUILD_EXE) -o $(TMPDIR)/exe.o
	$(ASSERT_IS_BITCODE_OBJ) $(TMPDIR)/exe.o
