//@ compile-flags: -C no-prepopulate-passes -Z mir-opt-level=0

#![crate_type = "lib"]
#![feature(static_align, const_attr_paths)]

const LOW: usize = 32;

mod alignments {
    pub const HIGH: usize = 64;
}

// CHECK: @CONST_ALIGN =
// CHECK-SAME: align 64
#[no_mangle]
#[rustc_align_static(LOW)]
#[rustc_align_static(alignments::HIGH)]
pub static CONST_ALIGN: u64 = 0;
