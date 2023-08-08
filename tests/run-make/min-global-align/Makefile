include ../tools.mk

# only-linux

# This tests ensure that global variables respect the target minimum alignment.
# The three bools `STATIC_BOOL`, `STATIC_MUT_BOOL`, and `CONST_BOOL` all have
# type-alignment of 1, but some targets require greater global alignment.

SRC = min_global_align.rs
LL = $(TMPDIR)/min_global_align.ll

all:
# Most targets are happy with default alignment -- take i686 for example.
ifeq ($(filter x86,$(LLVM_COMPONENTS)),x86)
	$(RUSTC) --target=i686-unknown-linux-gnu --emit=llvm-ir $(SRC)
	[ "$$(grep -c 'align 1' "$(LL)")" -eq "3" ]
endif
# SystemZ requires even alignment for PC-relative addressing.
ifeq ($(filter systemz,$(LLVM_COMPONENTS)),systemz)
	$(RUSTC) --target=s390x-unknown-linux-gnu --emit=llvm-ir $(SRC)
	[ "$$(grep -c 'align 2' "$(LL)")" -eq "3" ]
endif
