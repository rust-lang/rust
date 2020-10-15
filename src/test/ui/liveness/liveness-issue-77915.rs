// Ensure inout asm! operands are marked as used by the liveness pass

// only-x86_64
// check-pass

#![feature(asm)]
#![allow(dead_code)]
#![deny(unused_variables)]

// Tests the single variable inout case
unsafe fn rep_movsb(mut dest: *mut u8, mut src: *const u8, mut n: usize) -> *mut u8 {
    while n != 0 {
        asm!(
            "rep movsb",
            inout("rcx") n,
            inout("rsi") src,
            inout("rdi") dest,
        );
    }
    dest
}

// Tests the split inout case
unsafe fn rep_movsb2(mut dest: *mut u8, mut src: *const u8, mut n: usize) -> *mut u8 {
    while n != 0 {
        asm!(
            "rep movsb",
            inout("rcx") n,
            inout("rsi") src => src,
            inout("rdi") dest,
        );
    }
    dest
}

fn main() {}
