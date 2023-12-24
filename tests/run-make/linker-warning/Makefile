include ../tools.mk

RUN_RUSTC := $(RUSTC_ORIGINAL) main.rs -o $(TMPDIR)/main -C linker=./fake-linker.sh

all:
	# Run rustc with our fake linker, and make sure it shows warnings
	$(RUN_RUSTC) -C link-arg=run_make_warn 2>&1 | $(CGREP) "warning: linker stderr: bar"

	# Make sure it shows stdout, but only when --verbose is passed
	$(RUN_RUSTC) -C link-arg=run_make_info --verbose 2>&1 | $(CGREP) "warning: linker stdout: foo"
	$(RUN_RUSTC) -C link-arg=run_make_info 2>&1 | $(CGREP) -v "warning: linker stdout: foo"

	# Make sure we short-circuit this new path if the linker exits with an error (so the diagnostic is less verbose)
	rm -f $(TMPDIR)/main
	$(RUN_RUSTC) -C link-arg=run_make_error 2>&1 | $(CGREP) "note: error: baz"
	! [ -e $(TMPDIR)/main ]

