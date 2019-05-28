# needs-matching-clang

# This test makes sure that cross-language inlining can be used in conjunction
# with profile-guided optimization. The test only tests that the whole workflow
# can be executed without anything crashing. It does not test whether PGO or
# xLTO have any specific effect on the generated code.

-include ../tools.mk

COMMON_FLAGS=-Copt-level=3 -Ccodegen-units=1

# LLVM doesn't support instrumenting binaries that use SEH:
# https://bugs.llvm.org/show_bug.cgi?id=41279
#
# Things work fine with -Cpanic=abort though.
ifdef IS_MSVC
COMMON_FLAGS+= -Cpanic=abort
endif

all: cpp-executable rust-executable

cpp-executable:
	$(RUSTC) -Clinker-plugin-lto=on \
	         -Cprofile-generate="$(TMPDIR)"/cpp-profdata \
	         -o "$(TMPDIR)"/librustlib-xlto.a \
	         $(COMMON_FLAGS) \
	         ./rustlib.rs
	$(CLANG) -flto=thin \
	         -fprofile-generate="$(TMPDIR)"/cpp-profdata \
	         -fuse-ld=lld \
	         -L "$(TMPDIR)" \
	         -lrustlib-xlto \
	         -o "$(TMPDIR)"/cmain \
	         -O3 \
	         ./cmain.c
	$(TMPDIR)/cmain
	# Postprocess the profiling data so it can be used by the compiler
	"$(LLVM_BIN_DIR)"/llvm-profdata merge \
		-o "$(TMPDIR)"/cpp-profdata/merged.profdata \
		"$(TMPDIR)"/cpp-profdata/default_*.profraw
	$(RUSTC) -Clinker-plugin-lto=on \
	         -Cprofile-use="$(TMPDIR)"/cpp-profdata/merged.profdata \
	         -o "$(TMPDIR)"/librustlib-xlto.a \
	         $(COMMON_FLAGS) \
	         ./rustlib.rs
	$(CLANG) -flto=thin \
	         -fprofile-use="$(TMPDIR)"/cpp-profdata/merged.profdata \
	         -fuse-ld=lld \
	         -L "$(TMPDIR)" \
	         -lrustlib-xlto \
	         -o "$(TMPDIR)"/cmain \
	         -O3 \
	         ./cmain.c

rust-executable:
	exit
	$(CLANG) ./clib.c -fprofile-generate="$(TMPDIR)"/rs-profdata -flto=thin -c -o $(TMPDIR)/clib.o -O3
	(cd $(TMPDIR); $(AR) crus ./libxyz.a ./clib.o)
	$(RUSTC) -Clinker-plugin-lto=on \
	         -Cprofile-generate="$(TMPDIR)"/rs-profdata \
	         -L$(TMPDIR) \
	         $(COMMON_FLAGS) \
	         -Clinker=$(CLANG) \
	         -Clink-arg=-fuse-ld=lld \
	         -o $(TMPDIR)/rsmain \
	         ./main.rs
	$(TMPDIR)/rsmain
	# Postprocess the profiling data so it can be used by the compiler
	"$(LLVM_BIN_DIR)"/llvm-profdata merge \
		-o "$(TMPDIR)"/rs-profdata/merged.profdata \
		"$(TMPDIR)"/rs-profdata/default_*.profraw
	$(CLANG) ./clib.c \
	         -fprofile-use="$(TMPDIR)"/rs-profdata/merged.profdata \
	         -flto=thin \
	         -c \
	         -o $(TMPDIR)/clib.o \
	         -O3
	rm "$(TMPDIR)"/libxyz.a
	(cd $(TMPDIR); $(AR) crus ./libxyz.a ./clib.o)
	$(RUSTC) -Clinker-plugin-lto=on \
	         -Cprofile-use="$(TMPDIR)"/rs-profdata/merged.profdata \
	         -L$(TMPDIR) \
	         $(COMMON_FLAGS) \
	         -Clinker=$(CLANG) \
	         -Clink-arg=-fuse-ld=lld \
	         -o $(TMPDIR)/rsmain \
	         ./main.rs
