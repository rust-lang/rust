// only-x86_64

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

unsafe fn pclmul(a: __m128i, b: __m128i) -> __m128i {
    let imm8 = 3;
    _mm_clmulepi64_si128(a, b, imm8) //~ ERROR argument 3 is required to be a constant
}

fn main() {}
