//! regression test for <https://github.com/rust-lang/rust/issues/18119>
const X: u8 = 1;
static Y: u8 = 1;
fn foo() {}

impl X {}
//~^ ERROR cannot find type `X` in this scope
impl Y {}
//~^ ERROR cannot find type `Y` in this scope
impl foo {}
//~^ ERROR cannot find type `foo` in this scope

fn main() {}
