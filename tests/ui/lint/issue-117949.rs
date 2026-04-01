// Regression test for issue #117949

//@ revisions: noopt opt opt_with_overflow_checks
//@ [noopt]compile-flags: -C opt-level=0 -Z deduplicate-diagnostics=yes
//@ [opt]compile-flags: -O
//@ [opt_with_overflow_checks]compile-flags: -C overflow-checks=on -O -Z deduplicate-diagnostics=yes
//@ build-fail
//@ ignore-pass (test tests codegen-time behaviour)


fn main() {
    format_args!("{}", 1 << 32); //~ ERROR: arithmetic operation will overflow
    format_args!("{}", 1 >> 32); //~ ERROR: arithmetic operation will overflow
    format_args!("{}", 1 + i32::MAX); //~ ERROR: arithmetic operation will overflow
    format_args!("{}", -5 - i32::MAX); //~ ERROR: arithmetic operation will overflow
    format_args!("{}", 5 * i32::MAX); //~ ERROR: arithmetic operation will overflow
    format_args!("{}", 1 / 0); //~ ERROR: this operation will panic at runtime
    format_args!("{}", 1 % 0); //~ ERROR: this operation will panic at runtime
    format_args!("{}", [1, 2, 3][4]); //~ ERROR: this operation will panic at runtime
}
