-include ../tools.mk

all:
	$(RUSTC) lib.rs -C gen-crate-map
	ln -nsf $(call DYLIB,boot-*) $(call DYLIB,boot)
	$(CC) main.c -o $(call RUN,main) -lboot
	$(call RUN,main)
	rm $(call DYLIB,boot)
	$(call FAIL,main)
