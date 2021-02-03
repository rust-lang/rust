# Common Makefile include for Rust `run-make-fulldeps/coverage-* tests. Include this
# file with the line:
#
# -include ../coverage/coverage_tools.mk

-include ../tools.mk

# ISSUE(76038): When targeting MSVC, Rust binaries built with both `-Z instrument-coverage` and
# `-C link-dead-code` typically crash (with a seg-fault) or at best generate an empty `*.profraw`
# file, required for coverage reports.
#
# Enabling `-C link-dead-code` is not necessary when compiling with `-Z instrument-coverage`,
# due to improvements in the coverage map generation, to add unreachable functions known to Rust.
# Therefore, `-C link-dead-code` is no longer automatically enabled.

UNAME = $(shell uname)
