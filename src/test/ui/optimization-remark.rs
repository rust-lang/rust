// build-pass
// ignore-pass
// no-system-llvm
// revisions: all inline
//          compile-flags: --crate-type=lib -Cdebuginfo=1 -Copt-level=2
// [all]    compile-flags: -Cremark=all
// [inline] compile-flags: -Cremark=inline
// error-pattern: inline: f not inlined into g
// dont-check-compiler-stderr

#[no_mangle]
#[inline(never)]
pub fn f() {
}

#[no_mangle]
pub fn g() {
    f();
}
