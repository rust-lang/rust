define_type! { struct f64x8([f64; 8]); }

/*
#[cfg(target_arch = "x86")]
from_aligned! { unsafe f64x8 |bidirectional| core::arch::x86::__m512d }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe f64x8 |bidirectional| core::arch::x86_64::__m512d }
*/
