define_type! {
    #[doc = "Vector of two `f32` values"]
    struct f32x2([f32; 2]);
}

define_type! {
    #[doc = "Vector of four `f32` values"]
    struct f32x4([f32; 4]);
}

define_type! {
    #[doc = "Vector of eight `f32` values"]
    struct f32x8([f32; 8]);
}

define_type! {
    #[doc = "Vector of 16 `f32` values"]
    struct f32x16([f32; 16]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe f32x4 |bidirectional| core::arch::x86::__m128 }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe f32x4 |bidirectional| core::arch::x86_64::__m128 }

#[cfg(target_arch = "x86")]
from_aligned! { unsafe f32x8 |bidirectional| core::arch::x86::__m256 }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe f32x8 |bidirectional| core::arch::x86_64::__m256 }

/*
#[cfg(target_arch = "x86")]
from_aligned! { unsafe f32x16 |bidirectional| core::arch::x86::__m512 }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe f32x16 |bidirectional| core::arch::x86_64::__m512 }
*/
