//@ run-pass

// Regression test for <https://github.com/rust-lang/rust/issues/153596>.

#![allow(warnings)]

trait Trait {
    type Assoc;
}
impl<'a, 'b: 'a> Trait for Inv<'a, 'b> {
    type Assoc = ();
}

trait ReqWf {}
impl<T: Trait> ReqWf for T where T::Assoc: Sized {}
struct Inv<'a, 'b: 'a>(Option<*mut &'a &'b ()>);
fn mk_opaque<'a, 'b>(x: &'a &'b u32) -> impl ReqWf + use<'a, 'b> {
    Inv::<'a, 'b>(None)
}

trait Bound<T> {}
impl<T, F, R: ReqWf> Bound<T> for F where F: FnOnce(T) -> R {}
trait ImpossiblePredicates<F> {
    fn call_me(&self)
    where
        F: for<'a, 'b> Bound<&'a &'b u32>,
    {
        println!("method body");
    }
}
impl<F> ImpossiblePredicates<F> for () {}
fn mk_trait_object<F>(_: F) -> Box<dyn ImpossiblePredicates<F>> {
    Box::new(())
}
pub fn main() {
    let obj = mk_trait_object(mk_opaque);
    // This previously caused a segfault: the where-bounds of
    // `ImpossiblePredicate::call_me` did not hold due to missing implied bounds
    // for the fully normalized opaque type of `obj` in `fn impossible_predicates`.
    // As a result, the method's vtable ended up empty.
    //
    // However, earlier compilation passes did not report an error because the
    // opaque type had not yet been fully normalized.
    obj.call_me();
}
