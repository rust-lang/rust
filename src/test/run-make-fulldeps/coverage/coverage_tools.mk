# Common Makefile include for Rust `run-make-fulldeps/instrument-coverage-* tests. Include this
# file with the line:
#
# -include ../instrument-coverage/coverage_tools.mk
#
# To enable the Rust compiler option `-C link-dead-code`, also set the following variable
# *BEFORE* the `-include` line:
#
# LINK_DEAD_CODE=yes

-include ../tools.mk

ifndef LINK_DEAD_CODE
	LINK_DEAD_CODE=no
endif

# ISSUE(76038): When targeting MSVC, Rust binaries built with both `-Z instrument-coverage` and
# `-C link-dead-code` typically crash (with a seg-fault) or at best generate an empty `*.profraw`
# file, required for coverage reports.
#
# Enabling `-C link-dead-code` is preferred when compiling with `-Z instrument-coverage`, so
# `-C link-dead-code` is automatically enabled for all platform targets _except_ MSVC.
#
# Making the state of `-C link-dead-code` platform-dependent creates a problem for cross-platform
# tests because the injected counters, coverage reports, and some low-level output can be different,
# depending on the `-C link-dead-code` setting. For example, coverage reports will not report any
# coverage for a dead code region when the `-C link-dead-code` option is disabled, but with the
# option enabled, those same regions will show coverage counter values (of zero, of course).
#
# To ensure cross-platform `-Z instrument-coverage` generate consistent output, the
# `-C link-dead-code` option is always explicitly enabled or disabled.
#
# Since tests that execute binaries enabled with both `-Z instrument-coverage` and
# `-C link-dead-code` are known to fail, those tests will need the `# ignore-msvc` setting.
#
# If and when the above issue is resolved, the `# ignore-msvc` option can be removed, and the
# tests can be simplified to always test with `-C link-dead-code`.

UNAME = $(shell uname)

# FIXME(richkadel): Can any of the features tested by `run-make-fulldeps/coverage-*` tests be tested
# just as completely by more focused unit tests of the code logic itself, to reduce the number of
# test result files generated and maintained, and to help identify specific test failures and root
# causes more easily?
