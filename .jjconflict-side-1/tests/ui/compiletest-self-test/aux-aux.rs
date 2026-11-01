//@ aux-crate: aux_aux_foo=aux_aux_foo.rs
//@ aux-crate: aux_aux_bar=aux_aux_bar.rs
//@ edition: 2021
//@ compile-flags: --crate-type lib
//@ check-pass

use aux_aux_foo::Bar as IndirectBar;
use aux_aux_bar::Bar as DirectBar;

fn foo(x: IndirectBar) {}

fn main() {
    foo(DirectBar);
}
