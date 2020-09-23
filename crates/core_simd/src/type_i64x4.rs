define_type! {
    #[doc = "Vector of four `i64` types"]
    struct i64x4([i64; 4]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe i64x4 |bidirectional| core::arch::x86::__m256i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i64x4 |bidirectional| core::arch::x86_64::__m256i }
