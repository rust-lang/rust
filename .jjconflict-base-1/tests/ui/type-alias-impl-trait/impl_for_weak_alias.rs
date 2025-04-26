//! The orphan check expands a free alias used as the self type of an auto-trait impl, so an
//! alias resolving to a tuple is accepted just like the tuple written directly. See #157756.

//@ check-pass

#![feature(type_alias_impl_trait)]
#![feature(auto_traits)]

type Alias = (impl Sized, u8);

auto trait Trait {}
impl Trait for Alias {}

#[define_opaque(Alias)]
fn _def() -> Alias {
    (42, 42)
}

fn main() {}
