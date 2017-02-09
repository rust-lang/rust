-include ../tools.mk

all:
	$(RUSTC) -Z sanitizer=leak --target i686-unknown-linux-gnu hello.rs 2>&1 | grep -q 'Sanitizers only work with the `x86_64-unknown-linux-gnu` target'
