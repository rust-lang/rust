// check-pass

// FIXME(type_alias_impl_trait): What does this test? This needs a comment
// explaining what we're worried about here.
#![feature(type_alias_impl_trait)]
trait Trait {}
type Opaque<T> = impl Sized;
fn foo<T>() -> Opaque<T> {
    ()
}

impl<T, V> Trait for (T, V, V, u32) {}
impl<U, V> Trait for (Opaque<U>, V, i32, V) {}

fn main() {}
