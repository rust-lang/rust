//@ check-pass
//@ aux-build:empty-struct.rs

#[no_link]
//~^ WARN use of deprecated `no_link` attribute
//~| WARN this was previously accepted by the compiler
extern crate empty_struct;

fn main() {
    empty_struct::XEmpty {};
}
