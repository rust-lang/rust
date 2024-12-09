//! This is a regression test to ensure that we emit a diagnostic pointing to the
//! reason the type was rejected in inline assembly.

//@ only-x86_64

#![feature(repr_simd)]

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct Foo<const C: usize>([u8; C]);

pub unsafe fn foo<const C: usize>(a: Foo<C>) {
    std::arch::asm!(
        "movaps {src}, {src}",
        src = in(xmm_reg) a,
        //~^ ERROR: cannot use value of type `Foo<C>` for inline assembly
    );
}

fn main() {}
