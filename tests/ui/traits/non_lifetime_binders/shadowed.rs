#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

fn function<T>() where for<T> (): Sized {}
//~^ ERROR the name `T` is already used for a generic parameter

struct Struct<T>(T) where for<T> (): Sized;
//~^ ERROR the name `T` is already used for a generic parameter

impl<T> Struct<T> {
    fn method() where for<T> (): Sized {}
    //~^ ERROR the name `T` is already used for a generic parameter
}

fn repeated() where for<T, T> (): Sized {}
//~^ ERROR the name `T` is already used for a generic parameter

fn main() {}
