//! This is a regression test to ensure that we emit a diagnostic pointing to the
//! reason the type was rejected in inline assembly.

//@ only-x86_64

#![feature(repr_simd)]

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct Foo<const C: usize>([u8; C]);
//~^ ERROR: cannot evaluate SIMD vector length

pub unsafe fn foo<const C: usize>(a: Foo<C>) {
    std::arch::asm!(
        "movaps {src}, {src}",
        src = in(xmm_reg) a,
        //~^ NOTE: SIMD vector length needs to be known statically
    );
}

fn main() {}
