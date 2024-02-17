//@ unset-rustc-env:RUST_BACKTRACE
//@ compile-flags:-Z treat-err-as-bug=1
//@ error-pattern:stack backtrace:
//@ failure-status:101
//@ ignore-msvc
//@ normalize-stderr-test "note: .*" -> ""
//@ normalize-stderr-test "thread 'rustc' .*" -> ""
//@ normalize-stderr-test " +\d+:.*__rust_begin_short_backtrace.*" -> "(begin_short_backtrace)"
//@ normalize-stderr-test " +\d+:.*__rust_end_short_backtrace.*" -> "(end_short_backtrace)"
//@ normalize-stderr-test " +\d+:.*\n" -> ""
//@ normalize-stderr-test " +at .*\n" -> ""
//
// This test makes sure that full backtraces are used for ICEs when
// RUST_BACKTRACE is not set. It does this by checking for the presence of
// `__rust_{begin,end}_short_backtrace` markers, which only appear in full
// backtraces. The rest of the backtrace is filtered out.
//
// Ignored on msvc becaue the `__rust_{begin,end}_short_backtrace` symbols
// aren't reliable.

fn main() { missing_ident; }
