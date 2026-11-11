//@ check-pass

#![feature(return_type_notation)]

trait Foo {
    fn borrow(&mut self) -> impl Sized + '_;
}

fn live_past_borrow<T: Foo<borrow(..): 'static>>(mut t: T) {
    let x = t.borrow();
    drop(t);
    drop(x);
}

// Test that the `'_` item bound in `borrow` does not cause us to
// overlook the `'static` RTN bound.
fn overlapping_mut<T: Foo<borrow(..): 'static>>(mut t: T) {
    let x = t.borrow();
    let x = t.borrow();
}

fn main() {}
