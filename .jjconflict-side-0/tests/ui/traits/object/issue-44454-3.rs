// Taken from https://github.com/rust-lang/rust/issues/44454#issuecomment-1332781290

use std::any::Any;

trait Animal<X>: 'static {}

trait Projector {
    type Foo;
}

impl<X> Projector for dyn Animal<X> {
    type Foo = X;
}

fn make_static<'a, T>(t: &'a T) -> &'static T {
    let x: <dyn Animal<&'a T> as Projector>::Foo = t;
    let any = generic::<dyn Animal<&'a T>, &'a T>(x);
    //~^ ERROR: lifetime may not live long enough
    any.downcast_ref::<&'static T>().unwrap()
}

fn generic<T: Projector + Animal<U> + ?Sized, U>(x: <T as Projector>::Foo) -> Box<dyn Any> {
    make_static_any(x)
}

fn make_static_any<U: 'static>(u: U) -> Box<dyn Any> {
    Box::new(u)
}

fn main() {
    let a = make_static(&"salut".to_string());
    println!("{}", *a);
}
