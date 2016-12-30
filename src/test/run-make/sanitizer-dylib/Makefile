-include ../tools.mk

all:
	$(RUSTC) -Z sanitizer=leak --crate-type dylib hello.rs 2>&1 | grep -q 'Only executables and rlibs can be compiled with `-Z sanitizer`'
