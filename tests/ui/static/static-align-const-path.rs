//@ run-pass

#![feature(rustc_attrs, static_align, const_attr_paths)]

const ALIGN: usize = 128;

#[rustc_align_static(ALIGN)]
static ALIGNED: u64 = 0;

fn main() {
    assert_eq!((&raw const ALIGNED as usize) % ALIGN, 0);
}
