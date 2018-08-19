-include ../tools.mk

all:
	$(RUSTC) -O foo.rs
	$(RUSTC) -O -L $(TMPDIR) main.rs
