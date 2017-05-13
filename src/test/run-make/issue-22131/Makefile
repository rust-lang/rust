-include ../tools.mk

all: foo.rs
	$(RUSTC) --cfg 'feature="bar"' --crate-type lib foo.rs
	$(HOST_RPATH_ENV) '$(RUSTDOC)' --test --cfg 'feature="bar"' \
		-L $(TMPDIR) foo.rs |\
		grep -q 'test foo_0 ... ok'
