define_type! {
    #[doc = "Vector of four `f64` types"]
    struct f64x4([f64; 4]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe f64x4 |bidirectional| core::arch::x86::__m256d }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe f64x4 |bidirectional| core::arch::x86_64::__m256d }
