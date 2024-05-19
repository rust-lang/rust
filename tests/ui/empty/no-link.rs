//@ check-pass
//@ aux-build:empty-struct.rs

#[no_link]
extern crate empty_struct;

fn main() {
    empty_struct::XEmpty1 {};
}
