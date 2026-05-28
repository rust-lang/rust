#![feature(type_alias_impl_trait)]

type Opaque2<T> = impl Sized;
type Opaque<'a, T> = Opaque2<T>;
#[define_opaque(Opaque)]
fn defining<'a, T>(x: &'a i32) -> Opaque<T> { x }
//~^ ERROR: hidden type for `Opaque2<T>` captures lifetime that does not appear in bounds

fn main() {}
