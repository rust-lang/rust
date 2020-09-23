define_type! {
    #[doc = "Vector of 64 `u8` types"]
    struct u8x64([u8; 64]);
}

/*
#[cfg(target_arch = "x86")]
from_aligned! { unsafe u8x64 |bidirectional| core::arch::x86::__m512i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u8x64 |bidirectional| core::arch::x86_64::__m512i }
*/
