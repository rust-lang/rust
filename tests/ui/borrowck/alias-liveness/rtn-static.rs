// check-pass

#![feature(return_position_impl_trait_in_trait, return_type_notation)]
//~^ WARN the feature `return_type_notation` is incomplete

trait Foo {
    fn borrow(&mut self) -> impl Sized + '_;
}

// Test that the `'_` item bound in `borrow` does not cause us to
// overlook the `'static` RTN bound.
fn test<T: Foo<borrow(): 'static>>(mut t: T) {
    let x = t.borrow();
    let x = t.borrow();
}

fn main() {}
