define_type! { struct i32x8([i32; 8]); }

#[cfg(target_arch = "x86")]
from_aligned! { unsafe i32x8 |bidirectional| core::arch::x86::__m256i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i32x8 |bidirectional| core::arch::x86_64::__m256i }
