define_type! {
    #[doc = "Vector of two `i128` values"]
    struct i128x2([i128; 2]);
}

define_type! {
    #[doc = "Vector of four `i128` values"]
    struct i128x4([i128; 4]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe i128x2 |bidirectional| core::arch::x86::__m256i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i128x2 |bidirectional| core::arch::x86_64::__m256i }

/*
#[cfg(target_arch = "x86")]
from_aligned! { unsafe i128x4 |bidirectional| core::arch::x86::__m512i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i128x4 |bidirectional| core::arch::x86_64::__m512i }
*/
