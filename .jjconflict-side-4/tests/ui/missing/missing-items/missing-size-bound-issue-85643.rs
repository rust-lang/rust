// Regression test for issue https://github.com/rust-lang/rust/issues/85643
// where ?Sized bound is missing from suggestion

//@ aux-build: randmock.rs

extern crate randmock;

use randmock::*;

struct D;

impl TestTrait<()> for D {}
//~^ ERROR not all trait items implemented, missing: `test`
//~| HELP implement the missing item: `fn test<R>(&self, _: &mut R) where R: ?Sized, R: SecondTestTrait { todo!() }`
fn main() {}
