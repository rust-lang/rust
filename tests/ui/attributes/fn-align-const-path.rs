//@ run-pass
//@ ignore-wasm32 aligning functions is not currently supported on wasm (#143368)
//@ ignore-backends: gcc

#![feature(rustc_attrs, fn_align, const_attr_paths)]

const ALIGN: usize = 4096;

#[rustc_align(ALIGN)]
fn aligned() {}

fn main() {
    assert_eq!((aligned as fn() as usize & !1) % ALIGN, 0);
}
