# needs-sanitizer-support

-include ../tools.mk

# NOTE the address sanitizer only supports x86_64 linux and macOS

ifeq ($(TARGET),x86_64-apple-darwin)
EXTRA_RUSTFLAG=-C rpath
else
ifeq ($(TARGET),x86_64-unknown-linux-gnu)
EXTRA_RUSTFLAG=
endif
endif

all:
	$(RUSTC) -Z sanitizer=address --crate-type proc-macro --target $(TARGET) hello.rs 2>&1 | $(CGREP) '-Z sanitizer'
