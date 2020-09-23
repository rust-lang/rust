define_type! {
    #[doc = "Vector of eight `f32` types"]
    struct f32x8([f32; 8]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe f32x8 |bidirectional| core::arch::x86::__m256 }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe f32x8 |bidirectional| core::arch::x86_64::__m256 }
