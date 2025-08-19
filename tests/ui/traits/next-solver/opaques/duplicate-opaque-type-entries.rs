//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass
#![crate_type = "lib"]
trait Eq<T> {}
impl<T> Eq<T> for T {}
trait ConstrainAndEq<T> {}
impl<T, U> ConstrainAndEq<T> for U
where
    T: FnOnce() -> u32,
    U: FnOnce() -> u32,
    T: Eq<U>,
{}

fn constrain_and_eq<T: ConstrainAndEq<U>, U>(_: T, _: U) {}
fn foo<'a>() -> impl Sized + use<'a> {
    // This proves `foo<'a>: FnOnce() -> u32` and `foo<'1>: FnOnce() -> u32`,
    // We constrain both `opaque<'a>` and `opaque<'1>` to `u32`, resulting in
    // two distinct opaque type uses. Proving `foo<'a>: Eq<foo<'1>>` then
    // equates the two regions at which point the two opaque type keys are now
    // equal. This previously caused an ICE.
    constrain_and_eq(foo::<'a>, foo::<'_>);
    1u32
}
