-include ../tools.mk
all:
	$(RUSTDOC) -w html -o $(TMPDIR)/doc foo.rs
	cp verify.sh $(TMPDIR)
	$(call RUN,verify.sh) $(TMPDIR)
