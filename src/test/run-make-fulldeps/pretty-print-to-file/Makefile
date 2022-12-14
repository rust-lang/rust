include ../tools.mk

all:
	$(RUSTC) -o $(TMPDIR)/input.out -Zunpretty=normal input.rs
	diff -u $(TMPDIR)/input.out input.pp
