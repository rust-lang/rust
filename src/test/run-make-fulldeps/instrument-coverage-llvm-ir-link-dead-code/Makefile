# needs-profiler-support
# ignore-msvc

# LINK_DEAD_CODE requires ignore-msvc due to Issue #64685
LINK_DEAD_CODE=yes

-include ../instrument-coverage-llvm-ir-base/Makefile

# ISSUE(64685): When targeting MSVC, Rust binaries built with both `-Z instrument-coverage` and
# `-C link-dead-code` typically crash (with a seg-fault) or at best generate an empty `*.profraw`.
# See ../instrument-coverage/coverage_tools.mk for more information.