-include ../tools.mk

all: lib.rs
	$(HOST_RPATH_ENV) $(RUSTC) lib.rs
	$(HOST_RPATH_ENV) $(RUSTDOC) -w html -o $(TMPDIR)/doc lib.rs
	$(HOST_RPATH_ENV) $(RUSTDOC) -L $(TMPDIR) -w html -o $(TMPDIR)/doc user.rs
	$(HTMLDOCCK) $(TMPDIR)/doc lib.rs
	$(HTMLDOCCK) $(TMPDIR)/doc user.rs
