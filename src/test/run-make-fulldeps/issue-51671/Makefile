-include ../tools.mk

# ignore-windows

all:
	$(RUSTC) --emit=obj app.rs
	nm $(TMPDIR)/app.o | $(CGREP) rust_begin_unwind
	nm $(TMPDIR)/app.o | $(CGREP) rust_eh_personality
	nm $(TMPDIR)/app.o | $(CGREP) rust_oom
