//@ compile-flags: -C no-prepopulate-passes -Z mir-opt-level=0 -Clink-dead-code
//@ edition: 2024
//@ ignore-wasm32 aligning functions is not currently supported on wasm (#143368)

#![crate_type = "lib"]
#![feature(rustc_attrs, fn_align, const_attr_paths)]

const LOW: usize = 32;

mod alignments {
    pub const HIGH: usize = 64;
}

// CHECK-LABEL: @const_align
// CHECK-SAME: align 64
#[unsafe(no_mangle)]
#[rustc_align(LOW)]
#[rustc_align(alignments::HIGH)]
pub fn const_align() {}
