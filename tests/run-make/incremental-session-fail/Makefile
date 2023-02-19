include ../../run-make-fulldeps/tools.mk

SESSION_DIR := $(TMPDIR)/session
OUTPUT_FILE := $(TMPDIR)/build-output

all:
	echo $(TMPDIR)
	# Make it so that rustc will fail to create a session directory.
	touch $(SESSION_DIR)
	# Check exit code is 1 for an error, and not 101 for ICE.
	$(RUSTC) foo.rs --crate-type=rlib -C incremental=$(SESSION_DIR) > $(OUTPUT_FILE) 2>&1; [ $$? -eq 1 ]
	$(CGREP) "could not create incremental compilation crate directory" < $(OUTPUT_FILE)
	# -v tests are fragile, hopefully this text won't change
	$(CGREP) -v "internal compiler error" < $(OUTPUT_FILE)
