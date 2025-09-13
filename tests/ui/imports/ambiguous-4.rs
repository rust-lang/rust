//@ aux-build: ../ambiguous-4-extern.rs

extern crate ambiguous_4_extern;

fn main() {
    ambiguous_4_extern::id(); //~ ERROR cannot find function `id` in crate `ambiguous_4_extern`
    //^ FIXME: `id` should be identified as an ambiguous item.
}
