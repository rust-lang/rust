-include ../tools.mk
all:
	$(HOST_RPATH_ENV) $(RUSTDOC) -w json -o $(TMPDIR)/doc.json foo.rs
	$(HOST_RPATH_ENV) $(RUSTDOC) -o $(TMPDIR)/doc $(TMPDIR)/doc.json
