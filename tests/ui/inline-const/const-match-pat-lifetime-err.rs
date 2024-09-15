#![feature(inline_const_pat)]

use std::marker::PhantomData;

#[derive(PartialEq, Eq)]
pub struct InvariantRef<'a, T: ?Sized>(&'a T, PhantomData<&'a mut &'a T>);

#[derive(PartialEq, Eq)]
pub struct CovariantRef<'a, T: ?Sized>(&'a T);

impl<'a, T: ?Sized> InvariantRef<'a, T> {
    pub const fn new(r: &'a T) -> Self {
        InvariantRef(r, PhantomData)
    }
}

impl<'a> InvariantRef<'a, ()> {
    pub const NEW: Self = InvariantRef::new(&());
}

impl<'a> CovariantRef<'a, ()> {
    pub const NEW: Self = CovariantRef(&());
}

fn match_invariant_ref<'a>() {
    let y = ();
    match InvariantRef::new(&y) {
        //~^ ERROR `y` does not live long enough [E0597]
        const { InvariantRef::<'a>::NEW } => (),
    }
}

fn match_covariant_ref<'a>() {
    // Unclear if we should error here (should we be able to subtype the type of
    // `y.0`), but using the associated const directly in the pattern also
    // errors.
    let y: (CovariantRef<'static, _>,) = (CovariantRef(&()),);
    //~^ ERROR lifetime may not live long enough
    match y.0 {
        const { CovariantRef::<'a>::NEW } => (),
    }
}

fn main() {
    match_invariant_ref();
    match_covariant_ref();
}
