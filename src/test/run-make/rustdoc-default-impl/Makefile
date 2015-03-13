-include ../tools.mk

all: foo.rs bar.rs
	$(RUSTC) foo.rs --crate-type lib
	$(HOST_RPATH_ENV) $(RUSTDOC) -w html -o $(TMPDIR)/doc bar.rs -L $(TMPDIR)
