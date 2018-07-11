-include ../tools.mk

all:
ifeq ($(PROFILER_SUPPORT),1)
ifndef IS_WINDOWS
	$(RUSTC) -g -Z pgo-gen="$(TMPDIR)/test.profraw" test.rs
	$(call RUN,test) || exit 1
	[ -e "$(TMPDIR)/test.profraw" ] || (echo "No .profraw file"; exit 1)
endif
endif
