
include ../tools.mk

# ignore windows due to libLLVM being present in PATH and the PATH and library path being the same
# (so fixing it is harder). See #57765 for context
ifndef IS_WINDOWS

# This test makes sure that the object files we generate are actually
# LLVM bitcode files (as used by linker LTO plugins) when compiling with
# -Clinker-plugin-lto.

# this only succeeds for bitcode files
ASSERT_IS_BITCODE_OBJ=("$(LLVM_BIN_DIR)"/llvm-bcanalyzer $(1))
EXTRACT_OBJS=(cd $(TMPDIR); rm -f ./*.o; "$(LLVM_BIN_DIR)"/llvm-ar x $(1))

BUILD_LIB=$(RUSTC) lib.rs -Copt-level=2 -Clinker-plugin-lto -Ccodegen-units=1
BUILD_EXE=$(RUSTC) main.rs -Copt-level=2 -Clinker-plugin-lto -Ccodegen-units=1 --emit=obj

all: staticlib staticlib-fat-lto staticlib-thin-lto rlib exe cdylib rdylib

staticlib: lib.rs
	$(BUILD_LIB) --crate-type=staticlib -o $(TMPDIR)/liblib.a
	$(call EXTRACT_OBJS, liblib.a)
	for file in $(TMPDIR)/liblib.*.rcgu.o; do $(call ASSERT_IS_BITCODE_OBJ, $$file); done

staticlib-fat-lto: lib.rs
	$(BUILD_LIB) --crate-type=staticlib -o $(TMPDIR)/liblib-fat-lto.a -Clto=fat
	$(call EXTRACT_OBJS, liblib-fat-lto.a)
	for file in $(TMPDIR)/liblib-fat-lto.*.rcgu.o; do $(call ASSERT_IS_BITCODE_OBJ, $$file); done

staticlib-thin-lto: lib.rs
	$(BUILD_LIB) --crate-type=staticlib -o $(TMPDIR)/liblib-thin-lto.a -Clto=thin
	$(call EXTRACT_OBJS, liblib-thin-lto.a)
	for file in $(TMPDIR)/liblib-thin-lto.*.rcgu.o; do $(call ASSERT_IS_BITCODE_OBJ, $$file); done

rlib: lib.rs
	$(BUILD_LIB) --crate-type=rlib -o $(TMPDIR)/liblib.rlib
	$(call EXTRACT_OBJS, liblib.rlib)
	for file in $(TMPDIR)/liblib.*.rcgu.o; do $(call ASSERT_IS_BITCODE_OBJ, $$file); done

cdylib: lib.rs
	$(BUILD_LIB) --crate-type=cdylib --emit=obj -o $(TMPDIR)/cdylib.o
	$(call ASSERT_IS_BITCODE_OBJ, $(TMPDIR)/cdylib.o)

rdylib: lib.rs
	$(BUILD_LIB) --crate-type=dylib --emit=obj -o $(TMPDIR)/rdylib.o
	$(call ASSERT_IS_BITCODE_OBJ, $(TMPDIR)/rdylib.o)

exe: lib.rs
	$(BUILD_EXE) -o $(TMPDIR)/exe.o
	$(call ASSERT_IS_BITCODE_OBJ, $(TMPDIR)/exe.o)

else

all:

endif
