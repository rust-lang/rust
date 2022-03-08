// revisions: migrate nll
//[nll]compile-flags: -Z borrowck=mir

// Since we are testing nll (and migration) explicitly as a separate
// revisions, don't worry about the --compare-mode=nll on this test.

// ignore-compare-mode-nll

//[nll] check-pass
//[migrate] check-fail

#![feature(generic_associated_types)]

trait Foo<T> {
    type Type<'a>
    where
        T: 'a;
}

impl<T> Foo<T> for () {
    type Type<'a> = ()
    where
        T: 'a;
}

fn foo<T>() {
    let _: for<'a> fn(<() as Foo<T>>::Type<'a>, &'a T) = |_, _| ();
    //[migrate]~^ the parameter type `T` may not live long enough
}

pub fn main() {}
