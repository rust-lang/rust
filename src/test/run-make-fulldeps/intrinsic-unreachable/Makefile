-include ../tools.mk

# ignore-windows
#
# Because of Windows exception handling, the code is not necessarily any shorter.
# https://github.com/llvm-mirror/llvm/commit/64b2297786f7fd6f5fa24cdd4db0298fbf211466

all:
	$(RUSTC) -O --emit asm exit-ret.rs
	$(RUSTC) -O --emit asm exit-unreachable.rs
	test `wc -l < $(TMPDIR)/exit-unreachable.s` -lt `wc -l < $(TMPDIR)/exit-ret.s`
