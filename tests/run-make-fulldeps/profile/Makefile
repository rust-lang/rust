# needs-profiler-support

include ../tools.mk

all:
	$(RUSTC) -g -Z profile test.rs
	$(call RUN,test) || exit 1
	[ -e "$(TMPDIR)/test.gcno" ] || (echo "No .gcno file"; exit 1)
	[ -e "$(TMPDIR)/test.gcda" ] || (echo "No .gcda file"; exit 1)
	$(RUSTC) -g -Z profile -Z profile-emit=$(TMPDIR)/abc/abc.gcda test.rs
	$(call RUN,test) || exit 1
	[ -e "$(TMPDIR)/abc/abc.gcda" ] || (echo "gcda file not emitted to defined path"; exit 1)
