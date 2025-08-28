//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// A regression test for an error in `redis` while working on #139587.
//
// We check for structural equality when adding defining uses of opaques.
// In this test one defining use had anonymized regions while the other
// one did not, causing an error.
struct W<T>(T);
fn constrain<F: FnOnce(T) -> R, T, R>(f: F) -> R {
    loop {}
}
fn foo<'a>(x: for<'b> fn(&'b ())) -> impl Sized + use<'a> {
    let mut r = constrain(foo::<'_>);
    r = W(x);
    r
}
fn main() {}
