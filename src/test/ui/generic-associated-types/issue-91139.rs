// check-fail

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
    //~^ the parameter type `T` may not live long enough
}

pub fn main() {}
