-include ../tools.mk

all:
	$(RUSTC) a.rs --cfg x -C prefer-dynamic
	$(RUSTC) b.rs -C prefer-dynamic
	$(RUSTC) a.rs --cfg y -C prefer-dynamic
	$(call RUN,b) 2>&1 | grep "undefined symbol"
