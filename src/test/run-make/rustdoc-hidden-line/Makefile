-include ../tools.mk

# FIXME ignore windows
ifndef IS_WINDOWS

all:
	@echo $(RUSTDOC)
	$(HOST_RPATH_ENV) $(RUSTDOC) --test foo.rs
	$(HOST_RPATH_ENV) $(RUSTDOC) -w html -o $(TMPDIR)/doc foo.rs
	$(HTMLDOCCK) $(TMPDIR)/doc foo.rs

else
all:

endif
