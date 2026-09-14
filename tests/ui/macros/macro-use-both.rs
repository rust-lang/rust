//@ check-pass
//@ aux-build:two_macros.rs
//@ reference: macro.decl.scope.macro_use.syntax
//@ reference: macro.decl.scope.macro_use.prelude

#[macro_use(macro_one, macro_two)]
extern crate two_macros;

pub fn main() {
    macro_one!();
    macro_two!();
}
