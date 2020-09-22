define_type! { struct i16x32([i16; 32]); }

/*
#[cfg(target_arch = "x86")]
from_aligned! { unsafe u8x32 |bidirectional| core::arch::x86::__m512i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u8x32 |bidirectional| core::arch::x86_64::__m512i }
*/
