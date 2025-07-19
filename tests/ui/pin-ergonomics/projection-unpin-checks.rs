//@ edition:2024
#![feature(pin_ergonomics, negative_impls)]
#![allow(incomplete_features)]

// This test verifies that a `&pin mut T` cannot be projected to a pinned
// reference field `&pin mut T.U` if neither `T: !Unpin` nor `U: Unpin`.

struct Foo<T>(T);
struct NotUnpinFoo<T>(T, std::marker::PhantomPinned);
struct NegUnpinFoo<T>(T);

impl<T> !Unpin for NegUnpinFoo<T> {}

fn unpin2not_unpin<T>(foo_mut: &pin mut Foo<T>, foo_const: &pin const Foo<T>) {
    let Foo(_x) = foo_mut; //~ ERROR the trait bound `Foo<T>: !Unpin` is not satisfied [E0277]
    let Foo(_x) = foo_const; //~ ERROR the trait bound `Foo<T>: !Unpin` is not satisfied [E0277]
    let Foo(_x) = (|| foo_mut)(); //~ ERROR the trait bound `Foo<T>: !Unpin` is not satisfied [E0277]
    let Foo(_x) = (|| foo_const)(); //~ ERROR the trait bound `Foo<T>: !Unpin` is not satisfied [E0277]
    let &mut _foo_mut = foo_mut; //~ ERROR mismatched types
    let &_foo_const = foo_const; //~ ERROR mismatched types
    // FIXME(pin_ergonomics): add these tests since `&pin` pattern is ready
    // let &pin mut _foo_mut = foo_mut; // error
    // let &pin mut _foo_const = foo_const; // error
}

fn unpin2unpin<T: Unpin>(foo_mut: &pin mut Foo<T>, foo_const: &pin const Foo<T>) {
    let Foo(_x) = foo_mut; // ok
    let Foo(_x) = foo_const; // ok
    let Foo(_x) = (|| foo_mut)(); // ok
    let Foo(_x) = (|| foo_const)(); // ok
    let &mut _foo_mut = foo_mut; //~ ERROR mismatched types
    let &_foo_const = foo_const; //~ ERROR mismatched types
    // FIXME(pin_ergonomics): add these tests since `&pin` pattern is ready
    // let &pin mut _foo_mut = foo_mut; // ok
    // let &pin mut _foo_const = foo_const; // ok
}

fn not_unpin2not_unpin<T>(foo_mut: &pin mut NotUnpinFoo<T>, foo_const: &pin const NotUnpinFoo<T>) {
    let NotUnpinFoo(_x, _) = foo_mut; //~ ERROR the trait bound `NotUnpinFoo<T>: !Unpin` is not satisfied [E0277]
    let NotUnpinFoo(_x, _) = foo_const; //~ ERROR the trait bound `NotUnpinFoo<T>: !Unpin` is not satisfied [E0277]
    let NotUnpinFoo(_x, _) = (|| foo_mut)(); //~ ERROR the trait bound `NotUnpinFoo<T>: !Unpin` is not satisfied [E0277]
    let NotUnpinFoo(_x, _) = (|| foo_const)(); //~ ERROR the trait bound `NotUnpinFoo<T>: !Unpin` is not satisfied [E0277]
    let &mut _foo_mut = foo_mut; //~ ERROR mismatched types
    let &_foo_const = foo_const; //~ ERROR mismatched types
    // FIXME(pin_ergonomics): add these tests since `&pin` pattern is ready
    // let &pin mut _foo_mut = foo_mut; // error
    // let &pin mut _foo_const = foo_const; // error
}

fn not_unpin2unpin<T: Unpin>(
    foo_mut: &pin mut NotUnpinFoo<T>,
    foo_const: &pin const NotUnpinFoo<T>,
) {
    let NotUnpinFoo(_x, _) = foo_mut; // ok
    let NotUnpinFoo(_x, _) = foo_const; // ok
    let NotUnpinFoo(_x, _) = (|| foo_mut)(); // ok
    let NotUnpinFoo(_x, _) = (|| foo_const)(); // ok
    let &mut _foo_mut = foo_mut; //~ ERROR mismatched types
    let &_foo_const = foo_const; //~ ERROR mismatched types
    // FIXME(pin_ergonomics): add these tests since `&pin` pattern is ready
    // let &pin mut _foo_mut = foo_mut; // ok
    // let &pin mut _foo_const = foo_const; // ok
}

fn neg_unpin2not_unpin<T>(foo_mut: &pin mut NegUnpinFoo<T>, foo_const: &pin const NegUnpinFoo<T>) {
    let NegUnpinFoo(_x) = foo_mut; // ok
    let NegUnpinFoo(_x) = foo_const; // ok
    let NegUnpinFoo(_x) = (|| foo_mut)(); // ok
    let NegUnpinFoo(_x) = (|| foo_const)(); // ok
    let &mut _foo_mut = foo_mut; //~ ERROR mismatched types
    let &_foo_const = foo_const; //~ ERROR mismatched types
    // FIXME(pin_ergonomics): add these tests since `&pin` pattern is ready
    // let &pin mut _foo_mut = foo_mut; // ok
    // let &pin mut _foo_const = foo_const; // ok
}

fn neg_unpin2unpin<T: Unpin>(
    foo_mut: &pin mut NegUnpinFoo<T>,
    foo_const: &pin const NegUnpinFoo<T>,
) {
    let NegUnpinFoo(_x) = foo_mut; // ok
    let NegUnpinFoo(_x) = foo_const; // ok
    let NegUnpinFoo(_x) = (|| foo_mut)(); // ok
    let NegUnpinFoo(_x) = (|| foo_const)(); // ok
    let &mut _foo_mut = foo_mut; //~ ERROR mismatched types
    let &_foo_const = foo_const; //~ ERROR mismatched types
    // FIXME(pin_ergonomics): add these tests since `&pin` pattern is ready
    // let &pin mut _foo_mut = foo_mut; // ok
    // let &pin mut _foo_const = foo_const; // ok
}

fn tuple_ref_mut_pat_and_pin_mut_of_tuple_mut_ty<'a, T>(
    (&mut x,): &'a pin mut (&'a mut NegUnpinFoo<T>,),
    //~^ ERROR the trait bound `&'a mut NegUnpinFoo<T>: !Unpin` is not satisfied in `(&'a mut NegUnpinFoo<T>,)` [E0277]
) -> &'a pin mut NegUnpinFoo<T> {
    x
}

fn tuple_ref_mut_pat_and_pin_mut_of_mut_tuple_ty<'a, T>(
    (&mut x,): &'a pin mut &'a mut (NegUnpinFoo<T>,),
    //~^ ERROR mismatched type
    //~| ERROR the trait bound `&'a mut (NegUnpinFoo<T>,): !Unpin` is not satisfied [E0277]
) -> &'a pin mut NegUnpinFoo<T> {
    x
}

fn ref_mut_tuple_pat_and_pin_mut_of_tuple_mut_ty<'a, T>(
    &mut (x,): &'a pin mut (&'a mut NegUnpinFoo<T>,), //~ ERROR mismatched type
) -> &'a pin mut NegUnpinFoo<T> {
    x
}

fn ref_mut_tuple_pat_and_pin_mut_of_mut_tuple_ty<'a, T>(
    &mut (x,): &'a pin mut &'a mut (NegUnpinFoo<T>,), //~ ERROR mismatched type
) -> &'a pin mut NegUnpinFoo<T> {
    x
}

fn tuple_pat_and_pin_mut_of_tuple_mut_ty<'a, T>(
    (x,): &'a pin mut (&'a mut NegUnpinFoo<T>,),
) -> &'a pin mut &'a mut NegUnpinFoo<T> {
    x // ok
}

fn tuple_pat_and_pin_mut_of_mut_tuple_ty<'a, T>(
    (x,): &'a pin mut &'a mut (NegUnpinFoo<T>,),
    //~^ ERROR the trait bound `&'a mut (NegUnpinFoo<T>,): !Unpin` is not satisfied [E0277]
) -> &'a pin mut NegUnpinFoo<T> {
    x
}

fn main() {}
