-include ../tools.mk

all:
	$(RUSTC) lib.rs

	# the compiler needs to copy and modify the rlib file when performing
	# LTO, so we should ensure that it can cope with the original rlib
	# being read-only.
	chmod 444 $(TMPDIR)/*.rlib

	$(RUSTC) main.rs -C lto
	$(call RUN,main)
