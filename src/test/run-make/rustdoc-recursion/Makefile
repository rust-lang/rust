-include ../tools.mk

# FIXME ignore windows
ifndef IS_WINDOWS
all:
	$(HOST_RPATH_ENV) $(RUSTDOC) -w html -o $(TMPDIR)/doc foo.rs
	$(HOST_RPATH_ENV) $(RUSTDOC) -w html -o $(TMPDIR)/doc foo2.rs
	$(HOST_RPATH_ENV) $(RUSTDOC) -w html -o $(TMPDIR)/doc foo3.rs
else
all:
endif

