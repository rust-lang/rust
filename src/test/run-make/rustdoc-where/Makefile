-include ../tools.mk

all: foo.rs
	$(HOST_RPATH_ENV) $(RUSTDOC) -w html -o $(TMPDIR)/doc foo.rs
	$(HTMLDOCCK) $(TMPDIR)/doc foo.rs

