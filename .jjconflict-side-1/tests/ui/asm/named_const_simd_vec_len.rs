//! This is a regression test to ensure that we evaluate
//! SIMD vector length constants instead of assuming they are literals.

//@ only-x86_64
//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(repr_simd)]

const C: usize = 16;

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct Foo([u8; C]);

pub unsafe fn foo(a: Foo) {
    std::arch::asm!(
        "movaps {src}, {src}",
        src = in(xmm_reg) a,
    );
}

fn main() {}
