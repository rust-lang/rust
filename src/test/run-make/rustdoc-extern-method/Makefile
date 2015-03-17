-include ../tools.mk

all: foo.rs bar.rs
	$(HOST_RPATH_ENV) $(RUSTC) foo.rs
	$(HOST_RPATH_ENV) $(RUSTDOC) -w html -o $(TMPDIR)/doc foo.rs
	$(HOST_RPATH_ENV) $(RUSTDOC) -L $(TMPDIR) -w html -o $(TMPDIR)/doc bar.rs
	$(HTMLDOCCK) $(TMPDIR)/doc bar.rs
