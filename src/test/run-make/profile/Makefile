-include ../tools.mk

all:
ifeq ($(PROFILER_SUPPORT),1)
	$(RUSTC) -g -C profile test.rs
	$(call RUN,test) || exit 1
	[ -e "$(TMPDIR)/test.gcno" ] || (echo "No .gcno file"; exit 1)
	[ -e "$(TMPDIR)/test.gcda" ] || (echo "No .gcda file"; exit 1)
endif
