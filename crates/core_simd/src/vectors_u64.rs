define_type! {
    #[doc = "Vector of two `u64` values"]
    struct u64x2([u64; 2]);
}

define_type! {
    #[doc = "Vector of four `u64` values"]
    struct u64x4([u64; 4]);
}

define_type! {
    #[doc = "Vector of eight `u64` values"]
    struct u64x8([u64; 8]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe u64x2 |bidirectional| core::arch::x86::__m128i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u64x2 |bidirectional| core::arch::x86_64::__m128i }

#[cfg(target_arch = "x86")]
from_aligned! { unsafe u64x4 |bidirectional| core::arch::x86::__m256i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u64x4 |bidirectional| core::arch::x86_64::__m256i }

/*
#[cfg(target_arch = "x86")]
from_aligned! { unsafe u64x8 |bidirectional| core::arch::x86::__m512i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u64x8 |bidirectional| core::arch::x86_64::__m512i }
*/
