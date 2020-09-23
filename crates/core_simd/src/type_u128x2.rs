define_type! {
    #[doc = "Vector of two `u128` types"]
    struct u128x2([u128; 2]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe u128x2 |bidirectional| core::arch::x86::__m256i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u128x2 |bidirectional| core::arch::x86_64::__m256i }
