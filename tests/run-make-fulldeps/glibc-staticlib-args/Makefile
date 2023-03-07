# only-gnu
# only-linux

include ../tools.mk

# This ensures that std::env::args works in a library called from C on glibc Linux.

all:
	$(RUSTC) --crate-type=staticlib library.rs
	$(CC) program.c $(call STATICLIB,library) $(call OUT_EXE,program) \
		$(EXTRACFLAGS) $(EXTRACXXFLAGS)
	$(call RUN,program)
