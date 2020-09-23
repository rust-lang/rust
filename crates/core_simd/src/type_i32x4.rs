define_type! {
    #[doc = "Vector of four `i32` types"]
    struct i32x4([i32; 4]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe i32x4 |bidirectional| core::arch::x86::__m128i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i32x4 |bidirectional| core::arch::x86_64::__m128i }
