define_type! {
    #[doc = "Vector of eight `u16` types"]
    struct u16x8([u16; 8]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe u16x8 |bidirectional| core::arch::x86::__m128i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u16x8 |bidirectional| core::arch::x86_64::__m128i }
