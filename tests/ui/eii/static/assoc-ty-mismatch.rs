// Regression test for https://github.com/rust-lang/rust/issues/165139

//@ compile-flags: -Znext-solver
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu

#![feature(extern_item_impls)]

#[eii(hello)]
static HELLO: u8;

struct Dummy;
trait Trait {
    type Assoc;
}
impl Trait for Dummy {
    type Assoc = u16;
}

#[hello]
static HELLO_IMPL: <Dummy as Trait>::Assoc = 0_u16;
//~^ ERROR static `HELLO_IMPL` has a type that is incompatible with the declaration of `#[hello]`

fn main() {}
