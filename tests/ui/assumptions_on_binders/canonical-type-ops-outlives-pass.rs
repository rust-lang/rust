//@ check-pass
//@ revisions: default full next
//@[full] compile-flags: -Zassumptions-on-binders -Znext-solver
//@[next] compile-flags: -Znext-solver

// Exercise bound discharge and successive type operations in the same body.
#![allow(dead_code)]
trait Trait {
    type Assoc;
}
trait Bounded {
    type Assoc: 'static;
}
fn outlives<'a, T: 'a>() {}
fn environment<'a, T: Trait>()
where
    T::Assoc: 'a,
{
    outlives::<'a, T::Assoc>();
    outlives::<'a, T::Assoc>();
}
fn item_bound<T: Bounded>() {
    outlives::<'static, T::Assoc>();
}
fn component<'a, T: Trait + 'a>() {
    outlives::<'a, T::Assoc>();
}
fn implied<'a, T: Trait>(_: &'a T::Assoc) {
    outlives::<'a, T::Assoc>();
}
trait Object {
    fn method(&self) {}
}
impl<T> Object for T {}
fn body_bound<T: Trait>(value: T::Assoc) {
    let object: &dyn Object = &value;
    object.method();
}
trait Higher<'a> {
    type Assoc: 'static;
}
trait Check<'a, 'b> {}
impl<'a, 'b, T: Higher<'b>> Check<'a, 'b> for T where T::Assoc: 'a {}
trait Outer<'a> {}
impl<'a, T> Outer<'a> for T where T: for<'b> Check<'a, 'b> {}
fn require<T: for<'a> Outer<'a>>() {}
fn nested<T: for<'a> Higher<'a>>() {
    require::<T>();
}
trait Project {
    type Out;
}
impl<T: Trait> Project for (T,)
where
    T::Assoc: 'static,
{
    type Out = ();
}
fn normalize<T: Trait>()
where
    T::Assoc: 'static,
{
    let _: <(T,) as Project>::Out = ();
}
fn main() {}
