define_type! { struct i8x16([i8; 16]); }

#[cfg(target_arch = "x86")]
from_aligned! { unsafe i8x16 |bidirectional| core::arch::x86::__m128i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i8x16 |bidirectional| core::arch::x86_64::__m128i }
