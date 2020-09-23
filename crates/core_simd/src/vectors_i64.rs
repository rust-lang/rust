define_type! {
    #[doc = "Vector of two `i64` values"]
    struct i64x2([i64; 2]);
}

define_type! {
    #[doc = "Vector of four `i64` values"]
    struct i64x4([i64; 4]);
}

define_type! {
    #[doc = "Vector of eight `i64` values"]
    struct i64x8([i64; 8]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe i64x2 |bidirectional| core::arch::x86::__m128i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i64x2 |bidirectional| core::arch::x86_64::__m128i }

#[cfg(target_arch = "x86")]
from_aligned! { unsafe i64x4 |bidirectional| core::arch::x86::__m256i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i64x4 |bidirectional| core::arch::x86_64::__m256i }

/*
#[cfg(target_arch = "x86")]
from_aligned! { unsafe i64x8 |bidirectional| core::arch::x86::__m512i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i64x8 |bidirectional| core::arch::x86_64::__m512i }
*/
