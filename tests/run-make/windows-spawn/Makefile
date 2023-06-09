include ../tools.mk

# only-windows

all:
	$(RUSTC) -o "$(TMPDIR)/hopefullydoesntexist bar.exe" hello.rs
	$(RUSTC) spawn.rs
	$(TMPDIR)/spawn.exe
