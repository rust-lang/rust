-include ../tools.mk

all:
	# The tests must pass...
	$(RUSTC) main.rs
	$(call RUN,main)
	# ... and the loads/stores must not be optimized out.
	$(RUSTC) main.rs --emit=ir
	grep "load volatile"  $(TMPDIR)/main.ll
	grep "store volatile" $(TMPDIR)/main.ll
