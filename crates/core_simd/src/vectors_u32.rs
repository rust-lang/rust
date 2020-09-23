define_type! {
    #[doc = "Vector of two `u32` values"]
    struct u32x2([u32; 2]);
}

define_type! {
    #[doc = "Vector of four `u32` values"]
    struct u32x4([u32; 4]);
}

define_type! {
    #[doc = "Vector of eight `u32` values"]
    struct u32x8([u32; 8]);
}

define_type! {
    #[doc = "Vector of 16 `u32` values"]
    struct u32x16([u32; 16]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe u32x4 |bidirectional| core::arch::x86::__m128i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u32x4 |bidirectional| core::arch::x86_64::__m128i }

#[cfg(target_arch = "x86")]
from_aligned! { unsafe u32x8 |bidirectional| core::arch::x86::__m256i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u32x8 |bidirectional| core::arch::x86_64::__m256i }

/*
#[cfg(target_arch = "x86")]
from_aligned! { unsafe u32x16 |bidirectional| core::arch::x86::__m512i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u32x16 |bidirectional| core::arch::x86_64::__m512i }
*/
