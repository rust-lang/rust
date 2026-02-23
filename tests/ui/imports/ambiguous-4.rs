//@ edition:2015
//@ aux-build: ../ambiguous-4-extern.rs
//@ check-pass
extern crate ambiguous_4_extern;

fn main() {
    ambiguous_4_extern::id(); //~ WARN `id` is ambiguous
                              //~| WARN this was previously accepted
}
