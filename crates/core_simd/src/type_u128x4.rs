define_type! {
    #[doc = "Vector of four `u128` types"]
    struct u128x4([u128; 4]);
}

/*
#[cfg(target_arch = "x86")]
from_aligned! { unsafe u128x4 |bidirectional| core::arch::x86::__m512i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u128x4 |bidirectional| core::arch::x86_64::__m512i }
*/
