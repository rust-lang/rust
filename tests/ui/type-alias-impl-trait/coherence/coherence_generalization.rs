//@ check-pass

// FIXME(type_alias_impl_trait): What does this test? This needs a comment
// explaining what we're worried about here.

#![feature(type_alias_impl_trait)]
trait Trait {}
type Opaque<T> = impl Sized;
#[define_opaque(Opaque)]
fn foo<T>() -> Opaque<T> {
    ()
}

impl<T, U, V> Trait for (T, U, V, V, u32) {}
impl<U, V> Trait for (Opaque<U>, U, V, i32, V) {}

fn main() {}
