# needs-profiler-support

-include ../tools.mk

COMPILE_FLAGS=-O -Ccodegen-units=1 -Cprofile-generate="$(TMPDIR)"

# LLVM doesn't yet support instrumenting binaries that use unwinding on MSVC:
# https://github.com/rust-lang/rust/issues/61002
#
# Things work fine with -Cpanic=abort though.
ifdef IS_MSVC
COMPILE_FLAGS+= -Cpanic=abort
endif

all:
	$(RUSTC) $(COMPILE_FLAGS) --emit=llvm-ir test.rs
	# We expect symbols starting with "__llvm_profile_".
	$(CGREP) "__llvm_profile_" < $(TMPDIR)/test.ll
	# We do NOT expect the "__imp_" version of these symbols.
	$(CGREP) -v "__imp___llvm_profile_" < $(TMPDIR)/test.ll # 64 bit
	$(CGREP) -v "__imp____llvm_profile_" < $(TMPDIR)/test.ll # 32 bit
