#![feature(type_alias_impl_trait)]

type Opaque2<T> = impl Sized;
type Opaque<'a, T> = Opaque2<T>;
fn defining<'a, T>(x: &'a i32) -> Opaque<T> { x } //~ WARNING elided lifetime has a name
//~^ ERROR: hidden type for `Opaque2<T>` captures lifetime that does not appear in bounds

fn main() {}
