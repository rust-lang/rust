//! This test and `metasized.rs` and `pointeesized.rs` test that dyn-compatibility correctly
//! handles the different sizedness traits, which are special in several parts of the compiler.

trait Foo: std::fmt::Debug + Sized {}

impl<T: std::fmt::Debug + Sized> Foo for T {}

fn unsize_sized<T: 'static>(x: Box<T>) -> Box<dyn Sized> {
    //~^ ERROR the trait `Sized` is not dyn compatible
    x
}

fn unsize_subtrait(x: Box<dyn Foo>) -> Box<dyn Sized> {
    //~^ ERROR the trait `Foo` is not dyn compatible
    //~| ERROR the trait `Sized` is not dyn compatible
    x
}

fn main() {
    let _bx = unsize_sized(Box::new(vec![1, 2, 3]));
    //~^ ERROR the trait `Sized` is not dyn compatible

    let bx: Box<dyn Foo> = Box::new(vec![1, 2, 3]);
    //~^ ERROR the trait `Foo` is not dyn compatible
    let _ = format!("{bx:?}");
    let _bx = unsize_subtrait(bx);
    //~^ ERROR the trait `Foo` is not dyn compatible
    //~| ERROR the trait `Sized` is not dyn compatible
}
