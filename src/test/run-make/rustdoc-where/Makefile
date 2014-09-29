-include ../tools.mk

all: verify.sh foo.rs
	$(HOST_RPATH_ENV) $(RUSTDOC) -w html -o $(TMPDIR)/doc foo.rs
	cp verify.sh $(TMPDIR)
	$(call RUN,verify.sh) $(TMPDIR)
