include ../tools.mk

# ignore-windows

OUT := $(TMPDIR)/out

all: time

time: libc
	mkdir -p $(OUT)/time $(OUT)/time/deps
	ln -sf $(OUT)/libc/liblibc.rlib $(OUT)/time/deps/
	$(RUSTC) in/time/lib.rs -Ldependency=$(OUT)/time/deps/

libc:
	mkdir -p $(OUT)/libc
	$(RUSTC) in/libc/lib.rs --crate-name=libc -Cmetadata=foo -o $(OUT)/libc/liblibc.rlib
