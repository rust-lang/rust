define_type! {
    #[doc = "Vector of 32 `u16` types"]
    struct u16x32([u16; 32]);
}

/*
#[cfg(target_arch = "x86")]
from_aligned! { unsafe u16x32 |bidirectional| core::arch::x86::__m512i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u16x32 |bidirectional| core::arch::x86_64::__m512i }
*/
