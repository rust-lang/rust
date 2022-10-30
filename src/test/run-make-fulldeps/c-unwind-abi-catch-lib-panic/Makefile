include ../tools.mk

all: archive
	# Compile `main.rs`, which will link into our library, and run it.
	$(RUSTC) main.rs
	$(call RUN,main)

ifdef IS_MSVC
archive: add.o panic.o
	# Now, create an archive using these two objects.
	$(AR) crus $(TMPDIR)/add.lib $(TMPDIR)/add.o $(TMPDIR)/panic.o
else
archive: add.o panic.o
	# Now, create an archive using these two objects.
	$(AR) crus $(TMPDIR)/libadd.a $(TMPDIR)/add.o $(TMPDIR)/panic.o
endif

# Compile `panic.rs` into an object file.
#
# Note that we invoke `rustc` directly, so we may emit an object rather
# than an archive. We'll do that later.
panic.o:
	$(BARE_RUSTC) $(RUSTFLAGS)  \
		--out-dir $(TMPDIR) \
		--emit=obj panic.rs

# Compile `add.c` into an object file.
add.o:
	$(call COMPILE_OBJ,$(TMPDIR)/add.o,add.c)

