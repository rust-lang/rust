//@ known-bug: #134334
//@ only-x86_64

#[repr(simd)]
struct A();

fn main() {
    std::arch::asm!("{}", in(xmm_reg) A());
}
