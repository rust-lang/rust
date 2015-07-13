-include ../tools.mk

all:
	$(RUSTC) -C target-feature=-mmx,-sse,-sse2 ../../../libcore/lib.rs 2>&1 | \
		grep "LLVM ERROR: SSE register return with SSE disabled"
	$(RUSTC) --cfg disable_float -C target-feature=-mmx,-sse,-sse2 ../../../libcore/lib.rs
