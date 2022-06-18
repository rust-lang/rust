// run-pass

// Regression test for #55846, which once caused an ICE.

use std::marker::PhantomData;

struct Foo;

struct Bar<A> {
    a: PhantomData<A>,
}

impl Fooifier for Foo {
    type Assoc = Foo;
}

trait Fooifier {
    type Assoc;
}

trait Barifier<H> {
    fn barify();
}

impl<H> Barifier<H> for Bar<H> {
    fn barify() {
        println!("All correct!");
    }
}

impl Bar<<Foo as Fooifier>::Assoc> {
    fn this_shouldnt_crash() {
        <Self as Barifier<<Foo as Fooifier>::Assoc>>::barify();
    }
}

fn main() {
    Bar::<Foo>::this_shouldnt_crash();
}
