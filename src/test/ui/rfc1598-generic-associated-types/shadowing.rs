#![feature(generic_associated_types)]

//FIXME(#44265): The lifetime shadowing and type parameter shadowing
// should cause an error. Now it compiles (erroneously) and this will be addressed
// by a future PR. Then remove the following:
// build-pass (FIXME(62277): could be check-pass?)

trait Shadow<'a> {
    type Bar<'a>; // Error: shadowed lifetime
}

trait NoShadow<'a> {
    type Bar<'b>; // OK
}

impl<'a> NoShadow<'a> for &'a u32 {
    type Bar<'a> = i32; // Error: shadowed lifetime
}

trait ShadowT<T> {
    type Bar<T>; // Error: shadowed type parameter
}

trait NoShadowT<T> {
    type Bar<U>; // OK
}

impl<T> NoShadowT<T> for Option<T> {
    type Bar<T> = i32; // Error: shadowed type parameter
}

fn main() {}
