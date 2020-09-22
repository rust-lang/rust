define_type! { struct i128x2([i128; 2]); }

#[cfg(target_arch = "x86")]
from_aligned! { unsafe i128x2 |bidirectional| core::arch::x86::__m256i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe i128x2 |bidirectional| core::arch::x86_64::__m256i }
