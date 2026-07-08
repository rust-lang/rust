//@ compile-flags: -Znext-solver
// Test that anonymous const blocks using generic parameters are rejected
// under `generic_const_args`. Users should use named const items instead:
// `const FOO<const N: usize>: usize = N + 1;`
#![feature(generic_const_args, min_generic_const_args, generic_const_items)]

type const FOO<const N: usize>: usize = const { N + 1 }; //~ ERROR generic parameters in const blocks are not allowed; use a named `const` item instead

fn main() {}
