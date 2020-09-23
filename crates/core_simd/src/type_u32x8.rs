define_type! {
    #[doc = "Vector of eight `u32` types"]
    struct u32x8([u32; 8]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe u32x8 |bidirectional| core::arch::x86::__m256i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u32x8 |bidirectional| core::arch::x86_64::__m256i }
