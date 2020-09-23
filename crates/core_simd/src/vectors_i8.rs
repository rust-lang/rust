define_type! {
    #[doc = "Vector of two `i8` values"]
    struct i8x2([i8; 2]);
}

define_type! {
    #[doc = "Vector of four `i8` values"]
    struct i8x4([i8; 4]);
}

define_type! {
    #[doc = "Vector of eight `i8` values"]
    struct i8x8([i8; 8]);
}

define_type! {
    #[doc = "Vector of 16 `i8` values"]
    struct i8x16([i8; 16]);
}

define_type! {
    #[doc = "Vector of 32 `i8` values"]
    struct i8x32([i8; 32]);
}

define_type! {
    #[doc = "Vector of 64 `i8` values"]
    struct i8x64([i8; 64]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe i8x16 |bidirectional| core::arch::x86::__m128i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i8x16 |bidirectional| core::arch::x86_64::__m128i }

#[cfg(target_arch = "x86")]
from_aligned! { unsafe i8x32 |bidirectional| core::arch::x86::__m256i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i8x32 |bidirectional| core::arch::x86_64::__m256i }

/*
#[cfg(target_arch = "x86")]
from_aligned! { unsafe u8x64 |bidirectional| core::arch::x86::__m512i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u8x64 |bidirectional| core::arch::x86_64::__m512i }
*/
