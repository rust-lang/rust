#![crate_type = "lib"]

pub trait Foo<'a, T> {
    fn foo(&'a self) -> T;
}

pub fn foo<'a, T>(x: &'a Foo<'a, T>) -> T {
    let x: &'a Foo<T> = x;
    //            ^ the lifetime parameter of Foo is left to be inferred.
    x.foo()
    // ^ encoding this method call in metadata triggers an ICE.
}
