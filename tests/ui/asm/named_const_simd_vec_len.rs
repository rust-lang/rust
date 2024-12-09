//! This is a regression test to ensure that we evaluate
//! SIMD vector length constants instead of assuming they are literals.

#![feature(repr_simd)]

const C: usize = 16;

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct Foo([u8; C]);

pub unsafe fn foo(a: Foo) {
    std::arch::asm!(
        "movaps {src}, {src}",
        src = in(xmm_reg) a,
        //~^ ERROR: cannot use value of type `Foo` for inline assembly
    );
}

fn main() {}
