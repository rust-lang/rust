define_type! {
    #[doc = "Vector of 32 `i8` types"]
    struct i8x32([i8; 32]);
}

#[cfg(target_arch = "x86")]
from_aligned! { unsafe i8x32 |bidirectional| core::arch::x86::__m256i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i8x32 |bidirectional| core::arch::x86_64::__m256i }
