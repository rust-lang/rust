define_type! {
    #[doc = "Vector of two `u16` values"]
    struct u16x2([u16; 2]);
}

define_type! {
    #[doc = "Vector of four `u16` values"]
    struct u16x4([u16; 4]);
}

define_type! {
    #[doc = "Vector of eight `u16` values"]
    struct u16x8([u16; 8]);
}

define_type! {
    #[doc = "Vector of 16 `u16` values"]
    struct u16x16([u16; 16]);
}

define_type! {
    #[doc = "Vector of 32 `u16` values"]
    struct u16x32([u16; 32]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe u16x8 |bidirectional| core::arch::x86::__m128i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u16x8 |bidirectional| core::arch::x86_64::__m128i }

#[cfg(target_arch = "x86")]
from_aligned! { unsafe u16x16 |bidirectional| core::arch::x86::__m256i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u16x16 |bidirectional| core::arch::x86_64::__m256i }

/*
#[cfg(target_arch = "x86")]
from_aligned! { unsafe u16x32 |bidirectional| core::arch::x86::__m512i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u16x32 |bidirectional| core::arch::x86_64::__m512i }
*/
