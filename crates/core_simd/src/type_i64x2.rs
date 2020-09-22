define_type! { struct i64x2([i64; 2]); }

#[cfg(target_arch = "x86")]
from_aligned! { unsafe i64x2 |bidirectional| core::arch::x86::__m128i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i64x2 |bidirectional| core::arch::x86_64::__m128i }
