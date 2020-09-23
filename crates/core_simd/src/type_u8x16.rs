define_type! {
    #[doc = "Vector of 16 `u8` types"]
    struct u8x16([u8; 16]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe u8x16 |bidirectional| core::arch::x86::__m128i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u8x16 |bidirectional| core::arch::x86_64::__m128i }
