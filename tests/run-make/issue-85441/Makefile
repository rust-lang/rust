# only-windows-msvc

include ../tools.mk

# Tests that WS2_32.dll is not unnecessarily linked, see issue #85441

all:
	$(RUSTC) empty.rs
	objdump -p $(TMPDIR)/empty.exe | $(CGREP) -v -i "WS2_32.dll"
