-include ../tools.mk

# ignore-windows

all:
ifeq ($(PROFILER_SUPPORT),1)
	$(RUSTC) -g -Z pgo-gen="$(TMPDIR)/test.profraw" test.rs
	$(call RUN,test) || exit 1
	[ -e "$(TMPDIR)/test.profraw" ] || (echo "No .profraw file"; exit 1)
endif
