-include ../tools.mk

# FIXME ignore windows
ifndef IS_WINDOWS

source=index.rs

all:
	$(HOST_RPATH_ENV) $(RUSTDOC) -w html -o $(TMPDIR)/doc $(source)
	cp $(source) $(TMPDIR)
	cp verify.sh $(TMPDIR)
	$(call RUN,verify.sh) $(TMPDIR)

else
all:

endif
