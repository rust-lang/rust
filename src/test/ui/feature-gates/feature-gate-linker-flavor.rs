// This is a fake compile fail test as there's no way to generate a
// `#![feature(linker_flavor)]` error. The only reason we have a `linker_flavor`
// feature gate is to be able to document `-Z linker-flavor` in the unstable
// book

#[used]
fn foo() {}
//~^^ ERROR the `#[used]` attribute is an experimental feature

fn main() {}
