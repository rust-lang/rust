//@ check-pass

#![feature(type_alias_impl_trait)]
trait Foo {
    type Assoc;
}

impl Foo for i32 {
    type Assoc = u32;
}
type ImplTrait = impl Sized;
#[define_opaque(ImplTrait)]
fn constrain() -> ImplTrait {
    1u64
}
impl Foo for i64 {
    type Assoc = ImplTrait;
}

trait Bar<T> {}

impl<T: Foo> Bar<<T as Foo>::Assoc> for T {}

fn main() {}
