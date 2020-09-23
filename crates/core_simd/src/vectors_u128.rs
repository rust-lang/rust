define_type! {
    #[doc = "Vector of two `u128` values"]
    struct u128x2([u128; 2]);
}

define_type! {
    #[doc = "Vector of four `u128` values"]
    struct u128x4([u128; 4]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe u128x2 |bidirectional| core::arch::x86::__m256i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u128x2 |bidirectional| core::arch::x86_64::__m256i }

/*
#[cfg(target_arch = "x86")]
from_aligned! { unsafe u128x4 |bidirectional| core::arch::x86::__m512i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u128x4 |bidirectional| core::arch::x86_64::__m512i }
*/
