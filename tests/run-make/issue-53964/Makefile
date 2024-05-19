include ../tools.mk

all:
	$(RUSTC) panic.rs
	$(RUSTC) -C panic=abort --emit=obj app.rs -L $(TMPDIR)
