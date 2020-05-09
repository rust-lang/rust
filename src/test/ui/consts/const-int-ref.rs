// gate-test-const_int_refs
// revisions: gated vanilla

//[gated] check-pass

#![cfg_attr(gated, feature(const_int_ref))]

#[repr(C)]
union Transmute {
    int: usize,
    ptr: &'static i32
}

const GPIO: &i32 = unsafe { Transmute { int: 0x800 }.ptr };
//[vanilla]~^ ERROR it is undefined behavior

fn main() {}
