-include ../tools.mk

ifeq ($(TARGET),x86_64-unknown-linux-gnu)
all:
	$(RUSTC) -Z sanitizer=leak --crate-type dylib --target $(TARGET) hello.rs 2>&1 | grep -q 'Only executables and rlibs can be compiled with `-Z sanitizer`'
else
all:
endif
