define_type! {
    #[doc = "Vector of two `f64` values"]
    struct f64x2([f64; 2]);
}

define_type! {
    #[doc = "Vector of four `f64` values"]
    struct f64x4([f64; 4]);
}

define_type! {
    #[doc = "Vector of eight `f64` values"]
    struct f64x8([f64; 8]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe f64x2 |bidirectional| core::arch::x86::__m128d }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe f64x2 |bidirectional| core::arch::x86_64::__m128d }

#[cfg(target_arch = "x86")]
from_aligned! { unsafe f64x4 |bidirectional| core::arch::x86::__m256d }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe f64x4 |bidirectional| core::arch::x86_64::__m256d }

/*
#[cfg(target_arch = "x86")]
from_aligned! { unsafe f64x8 |bidirectional| core::arch::x86::__m512d }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe f64x8 |bidirectional| core::arch::x86_64::__m512d }
*/
