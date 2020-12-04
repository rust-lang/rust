# Common Makefile include for Rust `run-make-fulldeps/instrument-coverage-* tests. Include this
# file with the line:
#
# -include ../instrument-coverage/coverage_tools.mk

-include ../tools.mk

# ISSUE(76038): When targeting MSVC, Rust binaries built with both `-Z instrument-coverage` and
# `-C link-dead-code` typically crash (with a seg-fault) or at best generate an empty `*.profraw`
# file, required for coverage reports.
#
# Enabling `-C link-dead-code` is not necessary when compiling with `-Z instrument-coverage`,
# due to improvements in the coverage map generation, to add unreachable functions known to Rust.
# Therefore, `-C link-dead-code` is no longer automatically enabled.

UNAME = $(shell uname)

# Rust option `-Z instrument-coverage` uses LLVM Coverage Mapping Format version 4,
# which requires LLVM 11 or greater.
LLVM_VERSION_11_PLUS := $(shell \
		LLVM_VERSION=$$("$(LLVM_BIN_DIR)"/llvm-config --version) && \
		LLVM_VERSION_MAJOR=$${LLVM_VERSION/.*/} && \
		[ $$LLVM_VERSION_MAJOR -ge 11 ] && echo true || echo false)
