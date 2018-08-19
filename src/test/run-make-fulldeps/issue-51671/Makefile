-include ../tools.mk

ifdef IS_WINDOWS
# Do nothing on MSVC.
all:
	exit 0
else
all:
	$(RUSTC) --emit=obj app.rs
	nm $(TMPDIR)/app.o | $(CGREP) rust_begin_unwind
	nm $(TMPDIR)/app.o | $(CGREP) rust_eh_personality
	nm $(TMPDIR)/app.o | $(CGREP) rust_oom
endif
