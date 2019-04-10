-include ../tools.mk

all:
ifeq ($(PROFILER_SUPPORT),1)
	$(RUSTC) -Copt-level=3 -Clto=fat -Z pgo-gen="$(TMPDIR)" test.rs
	$(call RUN,test) || exit 1
	[ -e "$(TMPDIR)"/default_*.profraw ] || (echo "No .profraw file"; exit 1)
endif
