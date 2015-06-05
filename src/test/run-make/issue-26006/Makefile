-include ../tools.mk

ifndef IS_WINDOWS
all: time

time: libc
	mkdir -p out/time out/time/deps
	ln -sf out/libc/liblibc.rlib out/time/deps/
	$(RUSTC) in/time/lib.rs -Ldependency=out/time/deps/

libc:
	mkdir -p out/libc
	$(RUSTC) in/libc/lib.rs --crate-name=libc -o out/libc/liblibc.rlib
else
all:
endif
