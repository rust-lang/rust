define_type! {
    #[doc = "Vector of two `i32` values"]
    struct i32x2([i32; 2]);
}

define_type! {
    #[doc = "Vector of four `i32` values"]
    struct i32x4([i32; 4]);
}

define_type! {
    #[doc = "Vector of eight `i32` values"]
    struct i32x8([i32; 8]);
}

define_type! {
    #[doc = "Vector of 16 `i32` values"]
    struct i32x16([i32; 16]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe i32x4 |bidirectional| core::arch::x86::__m128i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i32x4 |bidirectional| core::arch::x86_64::__m128i }

#[cfg(target_arch = "x86")]
from_aligned! { unsafe i32x8 |bidirectional| core::arch::x86::__m256i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i32x8 |bidirectional| core::arch::x86_64::__m256i }

/*
#[cfg(target_arch = "x86")]
from_aligned! { unsafe u32x16 |bidirectional| core::arch::x86::__m512i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u32x16 |bidirectional| core::arch::x86_64::__m512i }
*/
