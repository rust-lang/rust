//@ check-pass

#![no_std]
#![crate_type = "lib"]
#![feature(strict_provenance_lints)]
#![warn(implicit_provenance_casts)]

pub fn int2ptr() -> *const u8 {
    16_usize as *const u8
    //~^ WARN cast from `usize` to `*const u8` implicitly relies on exposed provenance
}

pub fn ptr2int(ptr: *const u8) -> usize {
    ptr as usize
    //~^ WARN cast from `*const u8` to `usize` implicitly exposes pointer provenance
}
