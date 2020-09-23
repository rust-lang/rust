define_type! {
    #[doc = "Vector of four `i128` types"]
    struct i128x4([i128; 4]);
}

/*
#[cfg(target_arch = "x86")]
from_aligned! { unsafe i128x4 |bidirectional| core::arch::x86::__m512i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i128x4 |bidirectional| core::arch::x86_64::__m512i }
*/
