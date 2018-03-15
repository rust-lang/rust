-include ../tools.mk

all: foo.rs
	$(RUSTC) --cfg 'feature="bar"' --crate-type lib foo.rs
	$(RUSTDOC) --test --cfg 'feature="bar"' \
		-L $(TMPDIR) foo.rs |\
		grep -q 'foo.rs - foo (line 11) ... ok'
