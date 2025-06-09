#![crate_type = "lib"]

pub trait Foo<'a, T> {
    fn foo(&'a self) -> T;
}

pub fn foo<'a, T>(x: &'a dyn Foo<'a, T>) -> T {
    let x: &'a dyn Foo<T> = x;
    //                ^ the lifetime parameter of Foo is left to be inferred.
    x.foo()
    // ^ encoding this method call in metadata triggers an ICE.
}
