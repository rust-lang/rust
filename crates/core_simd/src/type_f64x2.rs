define_type! { struct f64x2([f64; 2]); }

#[cfg(target_arch = "x86")]
from_aligned! { unsafe f64x2 |bidirectional| core::arch::x86::__m128d }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe f64x2 |bidirectional| core::arch::x86_64::__m128d }
