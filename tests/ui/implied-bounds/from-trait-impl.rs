trait Trait {
    type Assoc;
}

impl<X: 'static> Trait for (X,) {
    type Assoc = ();
}

struct Foo<T: Trait>(T)
where
    T::Assoc: Clone; // any predicate using `T::Assoc` works here

fn func1(foo: Foo<(&str,)>) {
    //~^ ERROR `&str` does not fulfill the required lifetime
    //~| ERROR lifetime may not live long enough
    let _: &'static str = foo.0.0;
}

trait TestTrait {}

impl<X> TestTrait for [Foo<(X,)>; 1] {}
//~^ ERROR `X` may not live long enough
//~| ERROR `X` may not live long enough

fn main() {}
