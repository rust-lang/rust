define_type! { struct i16x16([i16; 16]); }

#[cfg(target_arch = "x86")]
from_aligned! { unsafe i16x16 |bidirectional| core::arch::x86::__m256i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i16x16 |bidirectional| core::arch::x86_64::__m256i }
