#![feature(type_alias_impl_trait)]

type Opaque<'a, T> = impl Sized;
#[defines(Opaque<'a, T>)]
fn defining<'a, T>(x: &'a i32) -> Opaque<T> {
    x
    //~^ ERROR: hidden type for `Opaque<'a, T>` captures lifetime that does not appear in bounds
}

fn main() {}
