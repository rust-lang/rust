// Regression test for issue #134224.

#![feature(repr_simd)]

#[repr(simd)]
struct A();
//~^ ERROR SIMD vector cannot be empty

fn main() {
    unsafe {
        std::arch::asm!("{}", in(xmm_reg) A());
        //~^ use of empty SIMD vector `A`
    }
}
