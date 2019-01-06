-include ../tools.mk

# ignore-windows
# ignore-macos

# Test for #39529.
# `-z text` causes ld to error if there are any non-PIC sections

all:
	$(RUSTC) hello.rs -C link-args=-Wl,-z,text
