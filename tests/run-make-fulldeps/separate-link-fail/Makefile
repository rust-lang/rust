include ../tools.mk

all:
	echo 'fn main(){}' > $(TMPDIR)/main.rs
	# Make sure that this fails
	! $(RUSTC) -Z link-only $(TMPDIR)/main.rs 2> $(TMPDIR)/stderr.txt
	$(CGREP) "The input does not look like a .rlink file" < $(TMPDIR)/stderr.txt
