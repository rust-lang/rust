-include ../tools.mk

# FIXME ignore windows
ifndef IS_WINDOWS

all:
	$(HOST_RPATH_ENV) $(RUSTDOC) --test foo.rs
	$(HOST_RPATH_ENV) $(RUSTDOC) -w html -o $(TMPDIR)/doc foo.rs
	cp verify.sh $(TMPDIR)
	$(call RUN,verify.sh) $(TMPDIR)

else
all:

endif
