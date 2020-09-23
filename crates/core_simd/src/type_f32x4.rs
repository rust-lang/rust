define_type! {
    #[doc = "Vector of four `f32` types"]
    struct f32x4([f32; 4]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe f32x4 |bidirectional| core::arch::x86::__m128 }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe f32x4 |bidirectional| core::arch::x86_64::__m128 }
