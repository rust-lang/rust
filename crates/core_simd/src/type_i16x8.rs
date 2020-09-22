define_type! { struct i16x8([i16; 8]); }

#[cfg(target_arch = "x86")]
from_aligned! { unsafe i16x8 |bidirectional| core::arch::x86::__m128i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i16x8 |bidirectional| core::arch::x86_64::__m128i }
