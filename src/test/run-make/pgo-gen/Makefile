-include ../tools.mk

all:
ifeq ($(PROFILER_SUPPORT),1)
	$(RUSTC) -g -C pgo-gen=test.profraw test.rs
	$(call RUN,test) || exit 1
	[ -e "$(TMPDIR)/test.profraw" ] || (echo "No .profraw file"; exit 1)
endif
