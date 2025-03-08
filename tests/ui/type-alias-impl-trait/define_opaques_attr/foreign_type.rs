#![feature(type_alias_impl_trait)]

#[define_opaques(String)]
//~^ ERROR: only opaque types defined in the local crate can be defined
fn main() {}
