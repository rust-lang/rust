//@ revisions: cfail1 cfail2
//@ build-pass

#![feature(const_attr_paths)]
#![allow(dead_code)]

#[cfg(cfail1)]
const ALIGN: usize = 8;
#[cfg(cfail2)]
const ALIGN: usize = 16;

#[cfg(cfail1)]
const EXPECT: usize = 8;
#[cfg(cfail2)]
const EXPECT: usize = 16;

#[repr(align(ALIGN))]
struct Aligned(u8);

const _: [(); EXPECT] = [(); core::mem::align_of::<Aligned>()];

fn main() {
    let _ = Aligned(0);
}
