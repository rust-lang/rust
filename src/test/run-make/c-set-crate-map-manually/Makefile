-include ../tools.mk

all:
	$(RUSTC) lib.rs -C gen-crate-map
	ln -nsf $(call DYLIB,boot-*) $(call DYLIB,boot)
	$(CC) main.c -o $(call RUN,main) -lboot -Wl,-rpath,$(TMPDIR)
	RUST_LOG=boot $(call RUN,main)
