//@ edition:2024

#![feature(const_trait_impl)]
struct Foo;
const impl Foo {
    async fn e() {}
    //~^ ERROR async functions are not allowed in `const` impls
}
fn main() {}
