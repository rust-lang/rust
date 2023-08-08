// build-pass
// ignore-pass
// revisions: all inline merge1 merge2
// compile-flags: --crate-type=lib -Cdebuginfo=1 -Copt-level=2
//
// Check that remarks can be enabled individually or with "all":
//
// [all]    compile-flags: -Cremark=all
// [inline] compile-flags: -Cremark=inline
//
// Check that values of -Cremark flag are accumulated:
//
// [merge1] compile-flags: -Cremark=all    -Cremark=giraffe
// [merge2] compile-flags: -Cremark=inline -Cremark=giraffe
//
// error-pattern: inline (missed): 'f' not inlined into 'g'
// dont-check-compiler-stderr

#[no_mangle]
#[inline(never)]
pub fn f() {
}

#[no_mangle]
pub fn g() {
    f();
}
