#[rustc_doc_primitive = "usize"]
//~^ ERROR use of an internal attribute [E0658]
//~| NOTE the `#[rustc_doc_primitive]` attribute is an internal implementation detail that will never be stable
//~| NOTE the `#[rustc_doc_primitive]` attribute is used by the standard library to provide a way to generate documentation for primitive types
/// Some docs
mod usize {}

fn main() {}
