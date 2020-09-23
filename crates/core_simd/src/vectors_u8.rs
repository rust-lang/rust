define_type! {
    #[doc = "Vector of two `u8` values"]
    struct u8x2([u8; 2]);
}

define_type! {
    #[doc = "Vector of four `u8` values"]
    struct u8x4([u8; 4]);
}

define_type! {
    #[doc = "Vector of eight `u8` values"]
    struct u8x8([u8; 8]);
}

define_type! {
    #[doc = "Vector of 16 `u8` values"]
    struct u8x16([u8; 16]);
}

define_type! {
    #[doc = "Vector of 32 `u8` values"]
    struct u8x32([u8; 32]);
}

define_type! {
    #[doc = "Vector of 64 `u8` values"]
    struct u8x64([u8; 64]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe u8x16 |bidirectional| core::arch::x86::__m128i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u8x16 |bidirectional| core::arch::x86_64::__m128i }

#[cfg(target_arch = "x86")]
from_aligned! { unsafe u8x32 |bidirectional| core::arch::x86::__m256i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u8x32 |bidirectional| core::arch::x86_64::__m256i }

/*
#[cfg(target_arch = "x86")]
from_aligned! { unsafe u8x64 |bidirectional| core::arch::x86::__m512i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u8x64 |bidirectional| core::arch::x86_64::__m512i }
*/
