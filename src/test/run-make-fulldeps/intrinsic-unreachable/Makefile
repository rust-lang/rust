-include ../tools.mk

ifndef IS_WINDOWS
# The assembly for exit-unreachable.rs should be shorter because it's missing
# (at minimum) a return instruction.

all:
	$(RUSTC) -O --emit asm exit-ret.rs
	$(RUSTC) -O --emit asm exit-unreachable.rs
	test `wc -l < $(TMPDIR)/exit-unreachable.s` -lt `wc -l < $(TMPDIR)/exit-ret.s`
else
# Because of Windows exception handling, the code is not necessarily any shorter.
# https://github.com/llvm-mirror/llvm/commit/64b2297786f7fd6f5fa24cdd4db0298fbf211466
all:
endif
