//@ build-pass
//@ force-host
//@ no-prefer-dynamic
//@ needs-unwind compiling proc macros with panic=abort causes a warning
//@ aux-build:exports_no_mangle.rs
#![crate_type = "proc-macro"]

// Issue #111888: this proc-macro crate imports another crate that itself
// exports a no_mangle function.
//
// That combination was broken for a period of time, because:
//
// In PR #99944 we *stopped* exporting no_mangle symbols from
// proc-macro crates. The constructed linker version script still referred
// to them, but resolving that discrepancy was left as a FIXME in the code.
//
// In PR #108017 we started telling the linker to check (via the
// `--no-undefined-version` linker invocation flag) that every symbol referenced
// in the "linker version script" is actually present in the linker input. So
// the unresolved discrepancy from #99944 started surfacing as a compile-time
// error.

extern crate exports_no_mangle;
