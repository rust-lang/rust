define_type! {
    #[doc = "Vector of two `i16` values"]
    struct i16x2([i16; 2]);
}

define_type! {
    #[doc = "Vector of four `i16` values"]
    struct i16x4([i16; 4]);
}

define_type! {
    #[doc = "Vector of eight `i16` values"]
    struct i16x8([i16; 8]);
}

define_type! {
    #[doc = "Vector of 16 `i16` values"]
    struct i16x16([i16; 16]);
}

define_type! {
    #[doc = "Vector of 32 `i16` values"]
    struct i16x32([i16; 32]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe i16x8 |bidirectional| core::arch::x86::__m128i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i16x8 |bidirectional| core::arch::x86_64::__m128i }

#[cfg(target_arch = "x86")]
from_aligned! { unsafe i16x16 |bidirectional| core::arch::x86::__m256i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i16x16 |bidirectional| core::arch::x86_64::__m256i }

/*
#[cfg(target_arch = "x86")]
from_aligned! { unsafe u8x32 |bidirectional| core::arch::x86::__m512i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u8x32 |bidirectional| core::arch::x86_64::__m512i }
*/
