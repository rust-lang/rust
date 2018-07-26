-include ../tools.mk

ifdef IS_WINDOWS

all:
	$(RUSTC) -o "$(TMPDIR)/hopefullydoesntexist bar.exe" hello.rs
	$(RUSTC) spawn.rs
	$(TMPDIR)/spawn.exe

else

all:

endif
