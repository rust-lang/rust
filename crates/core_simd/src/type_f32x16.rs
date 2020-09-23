define_type! {
    #[doc = "Vector of 16 `f32` types"]
    struct f32x16([f32; 16]);
}

/*
#[cfg(target_arch = "x86")]
from_aligned! { unsafe f32x16 |bidirectional| core::arch::x86::__m512 }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe f32x16 |bidirectional| core::arch::x86_64::__m512 }
*/
