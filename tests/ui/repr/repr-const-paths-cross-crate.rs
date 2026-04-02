//@ check-pass
//@ aux-build:repr_const_paths_aux.rs

#![feature(const_attr_paths)]

extern crate repr_const_paths_aux;

#[repr(align(repr_const_paths_aux::ALIGN))]
struct CrossCrate(u8);

fn main() {
    let _ = CrossCrate(0);
}
