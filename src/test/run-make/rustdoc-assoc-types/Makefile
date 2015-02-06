-include ../tools.mk

all: lib.rs
	$(HOST_RPATH_ENV) $(RUSTDOC) -w html -o $(TMPDIR)/doc lib.rs
	$(HTMLDOCCK) $(TMPDIR)/doc lib.rs
