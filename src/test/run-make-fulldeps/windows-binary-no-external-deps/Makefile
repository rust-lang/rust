-include ../tools.mk

# only-windows

PATH=$(SYSTEMROOT)/system32

all:
	$(RUSTC) hello.rs -C target-feature=+crt-static
	$(TMPDIR)/hello.exe
