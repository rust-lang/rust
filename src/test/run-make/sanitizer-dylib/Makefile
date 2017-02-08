-include ../tools.mk

all:
	$(RUSTC) -Z sanitizer=leak --crate-type dylib --target x86_64-unknown-linux-gnu hello.rs 2>&1 | grep -q 'Only executables and rlibs can be compiled with `-Z sanitizer`'
