//@ compile-flags: -Znext-solver
// Test that anonymous const blocks using generic parameters are rejected
// under `gca_const_items`. Users should use named const items instead:
// `const FOO<const N: usize>: usize = N + 1;`
#![feature(gca_const_items, gca_min_const_items, generic_const_items)]

use std::gca;

const FOO<const N: usize>: usize = gca!(const { N + 1 }); //~ ERROR generic parameters in const blocks are not allowed; use a named `const` item instead

fn main() {}
