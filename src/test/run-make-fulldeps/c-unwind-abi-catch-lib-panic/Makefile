-include ../tools.mk

all:
	# Compile `panic.rs` and `add.c` into object files.
	#
	# Note that we invoke `rustc` directly, so we may emit an object rather
	# than an archive. We'll do that later.
	$(BARE_RUSTC) $(RUSTFLAGS)  \
		--out-dir $(TMPDIR) \
		--emit=obj panic.rs
	$(call COMPILE_OBJ,$(TMPDIR)/add.o,add.c)

	# Now, create an archive using these two objects.
	$(AR) crus $(TMPDIR)/libadd.a $(TMPDIR)/add.o $(TMPDIR)/panic.o

	# Compile `main.rs`, which will link into our library, and run it.
	$(RUSTC) main.rs
	$(call RUN,main)
