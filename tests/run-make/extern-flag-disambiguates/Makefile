include ../tools.mk

# Attempt to build this dependency tree:
#
#	A.1   A.2
#	 |\    |
#	 | \   |
#        B  \  C
#         \ | /
#          \|/
#           D
#
# Note that A.1 and A.2 are crates with the same name.

all:
	$(RUSTC) -C metadata=1 -C extra-filename=-1 a.rs
	$(RUSTC) -C metadata=2 -C extra-filename=-2 a.rs
	$(RUSTC) b.rs --extern a=$(TMPDIR)/liba-1.rlib
	$(RUSTC) c.rs --extern a=$(TMPDIR)/liba-2.rlib
	@echo before
	$(RUSTC) --cfg before d.rs --extern a=$(TMPDIR)/liba-1.rlib
	$(call RUN,d)
	@echo after
	$(RUSTC) --cfg after  d.rs --extern a=$(TMPDIR)/liba-1.rlib
	$(call RUN,d)
