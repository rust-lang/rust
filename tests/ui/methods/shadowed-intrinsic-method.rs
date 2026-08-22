// Can't use rustfix because we provide two suggestions:
// to remove the arg for `Borrow::borrow` or to call `Type::borrow`.
use std::borrow::Borrow;

struct A;

impl A { fn borrow(&mut self, _: ()) {} }

struct B;

fn main() {
    // The fully-qualified path for items within functions is unnameable from outside that function.
    impl B { fn borrow(&mut self, _: ()) {} }

    struct C;
    // The fully-qualified path for items within functions is unnameable from outside that function.
    impl C { fn borrow(&mut self, _: ()) {} }

    let mut a = A;
    a.borrow(()); //~ ERROR E0061
    // A::borrow(&mut a, ());
    let mut b = B;
    b.borrow(()); //~ ERROR E0061
    // This currently suggests `main::<impl B>::borrow`, which is not correct, it should be
    // B::borrow(&mut b, ());
    let mut c = C;
    c.borrow(()); //~ ERROR E0061
    // This currently suggests `main::C::borrow`, which is not correct, it should be
    // C::borrow(&mut c, ());
}

fn foo() {
    let mut b = B;
    b.borrow(()); //~ ERROR E0061
    // This currently suggests `main::<impl B>::borrow`, which is not correct, it should be
    // B::borrow(&mut b, ());
}
