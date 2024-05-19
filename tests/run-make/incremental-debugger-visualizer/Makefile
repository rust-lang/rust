include ../tools.mk

# This test makes sure that changes to files referenced via #[debugger_visualizer]
# are picked up when compiling incrementally.

# We have to copy the source to $(TMPDIR) because Github CI mounts the source
# directory as readonly. We need to apply modifications to some of the source
# file.
SRC_DIR := $(TMPDIR)/src
INCR_CACHE_DIR := $(TMPDIR)/incremental

all:
	rm -rf $(TMPDIR)/*
	mkdir $(SRC_DIR)
	cp ./foo.rs $(SRC_DIR)
	echo "GDB script v1" > $(SRC_DIR)/foo.py
	echo "Natvis v1" > $(SRC_DIR)/foo.natvis
	$(RUSTC) $(SRC_DIR)/foo.rs \
	  --crate-type=rlib \
	  --emit metadata \
	  -C incremental=$(INCR_CACHE_DIR) \
	  -Z incremental-verify-ich
	$(CGREP) "GDB script v1" < $(TMPDIR)/libfoo.rmeta
	$(CGREP) "Natvis v1" < $(TMPDIR)/libfoo.rmeta

	# Change only the GDB script and check that the change has been picked up
	echo "GDB script v2" > $(SRC_DIR)/foo.py
	$(RUSTC) $(SRC_DIR)/foo.rs \
	  --crate-type=rlib \
	  --emit metadata \
	  -C incremental=$(INCR_CACHE_DIR) \
	  -Z incremental-verify-ich

	$(CGREP) "GDB script v2" < $(TMPDIR)/libfoo.rmeta
	$(CGREP) -v "GDB script v1" < $(TMPDIR)/libfoo.rmeta
	$(CGREP) "Natvis v1" < $(TMPDIR)/libfoo.rmeta

	# Now change the Natvis version and check that the change has been picked up
	echo "Natvis v2" > $(SRC_DIR)/foo.natvis
	$(RUSTC) $(SRC_DIR)/foo.rs \
	  --crate-type=rlib \
	  --emit metadata \
	  -C incremental=$(INCR_CACHE_DIR) \
	  -Z incremental-verify-ich

	$(CGREP) "GDB script v2" < $(TMPDIR)/libfoo.rmeta
	$(CGREP) -v "GDB script v1" < $(TMPDIR)/libfoo.rmeta
	$(CGREP) "Natvis v2" < $(TMPDIR)/libfoo.rmeta
	$(CGREP) -v "Natvis v1" < $(TMPDIR)/libfoo.rmeta
