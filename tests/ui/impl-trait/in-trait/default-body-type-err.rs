use std::ops::Deref;

pub trait Foo {
    fn lol(&self) -> impl Deref<Target = String> {
        //~^ type mismatch resolving `<&i32 as Deref>::Target == String`
        &1i32
    }
}

fn main() {}
