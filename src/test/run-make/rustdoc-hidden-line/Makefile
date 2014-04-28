-include ../tools.mk

# FIXME ignore windows
ifndef IS_WINDOWS

all:
	$(RUSTDOC) --test foo.rs
	$(RUSTDOC) -w html -o $(TMPDIR)/doc foo.rs
	cp verify.sh $(TMPDIR)
	$(call RUN,verify.sh) $(TMPDIR)

else
all:

endif
