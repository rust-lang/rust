# needs-profiler-support
# ignore-windows-gnu

# FIXME(mati865): MinGW GCC miscompiles compiler-rt profiling library but with Clang it works
# properly. Since we only have GCC on the CI ignore the test for now.

-include ../tools.mk

COMPILE_FLAGS=-g -Cprofile-generate="$(TMPDIR)"

# LLVM doesn't yet support instrumenting binaries that use unwinding on MSVC:
# https://github.com/rust-lang/rust/issues/61002
#
# Things work fine with -Cpanic=abort though.
ifdef IS_MSVC
COMPILE_FLAGS+= -Cpanic=abort
endif

all:
	$(RUSTC) $(COMPILE_FLAGS) test.rs
	$(call RUN,test) || exit 1
	[ -e "$(TMPDIR)"/default_*.profraw ] || (echo "No .profraw file"; exit 1)
