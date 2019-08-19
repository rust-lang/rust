#![allow(incomplete_features)]
#![feature(generic_associated_types)]

trait Shadow<'a> {
    //FIXME(#44265): The lifetime parameter shadowing should cause an error.
    type Bar<'a>;
}

trait NoShadow<'a> {
    type Bar<'b>; // OK
}

impl<'a> NoShadow<'a> for &'a u32 {
    //FIXME(#44265): The lifetime parameter shadowing should cause an error.
    type Bar<'a> = i32;
}

trait ShadowT<T> {
    type Bar<T>; //~ ERROR the name `T` is already used
}

trait NoShadowT<T> {
    type Bar<U>; // OK
}

impl<T> NoShadowT<T> for Option<T> {
    type Bar<T> = i32; //~ ERROR the name `T` is already used
}

fn main() {}
