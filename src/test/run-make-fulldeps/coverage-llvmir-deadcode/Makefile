# needs-profiler-support
# ignore-msvc

# LINK_DEAD_CODE requires ignore-msvc due to Issue #76038
LINK_DEAD_CODE=yes

-include ../coverage-llvmir-base/Makefile

# ISSUE(76038): When targeting MSVC, Rust binaries built with both `-Z instrument-coverage` and
# `-C link-dead-code` typically crash (with a seg-fault) or at best generate an empty `*.profraw`.
# See ../coverage/coverage_tools.mk for more information.
