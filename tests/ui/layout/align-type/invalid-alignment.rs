// test that invalid alignments put into Align fail
#![feature(align_type)]

use std::mem::Align;

const MAX_SUPPORTED_ALIGN: usize = 1 << 29;

const _: () = {
    // not power of two
    align_of::<Align<3>>(); //~? ERROR unknown layout
};

const _: () = {
    // too big
    align_of::<Align<{MAX_SUPPORTED_ALIGN * 2}>>(); //~? ERROR unknown layout
};

fn main() {}
