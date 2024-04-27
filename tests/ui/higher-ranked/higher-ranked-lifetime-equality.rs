// A regression test for #97156

type One = for<'a> fn(&'a (), &'a ());
type Two = for<'a, 'b> fn(&'a (), &'b ());

mod my_api {
    use std::any::Any;
    use std::marker::PhantomData;

    pub struct Foo<T: 'static> {
        a: &'static dyn Any,
        _p: PhantomData<*mut T>, // invariant, the type of the `dyn Any`
    }

    impl<T: 'static> Foo<T> {
        pub fn deref(&self) -> &'static T {
            match self.a.downcast_ref::<T>() {
                None => unsafe { std::hint::unreachable_unchecked() },
                Some(a) => a,
            }
        }

        pub fn new(a: T) -> Foo<T> {
            Foo::<T> { a: Box::leak(Box::new(a)), _p: PhantomData }
        }
    }
}

use my_api::*;

fn main() {
    let foo = Foo::<One>::new((|_, _| ()) as One);
    foo.deref();
    let foo: Foo<Two> = foo;
    //~^ ERROR mismatched types [E0308]
    //~| ERROR mismatched types [E0308]
    foo.deref();
}
