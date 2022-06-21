// check-pass

#![feature(type_alias_impl_trait)]
type Opaque<T> = impl Sized;
fn defining<T>() -> Opaque<T> {}
struct Ss<'a, T>(&'a Opaque<T>);

fn main() {}
