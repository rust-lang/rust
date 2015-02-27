-include ../tools.mk

all: lib.rs ext.rs
	$(HOST_RPATH_ENV) $(RUSTC) ext.rs
	$(HOST_RPATH_ENV) $(RUSTDOC) -L $(TMPDIR) -w html -o $(TMPDIR)/doc lib.rs
	$(HTMLDOCCK) $(TMPDIR)/doc lib.rs
