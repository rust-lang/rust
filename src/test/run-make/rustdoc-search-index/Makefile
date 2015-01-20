-include ../tools.mk

# FIXME ignore windows
ifndef IS_WINDOWS

source=index.rs

all:
	$(HOST_RPATH_ENV) $(RUSTDOC) -w html -o $(TMPDIR)/doc $(source)
	$(HTMLDOCCK) $(TMPDIR)/doc $(source)

else
all:

endif
