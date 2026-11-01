//@ reference: macro.decl.scope.macro_use.syntax
#![no_std]

#[macro_use(foo="bar")]  //~ ERROR malformed `macro_use` attribute input
extern crate std;

fn main() {}
