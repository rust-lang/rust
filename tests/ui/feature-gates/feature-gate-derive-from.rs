//@ edition: 2021

#[derive(From)] //~ ERROR use of unstable library feature `derive_from`
struct Foo(u32);

fn main() {}
