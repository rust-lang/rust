define_type! { struct i32x16([i32; 16]); }

/*
#[cfg(target_arch = "x86")]
from_aligned! { unsafe u32x16 |bidirectional| core::arch::x86::__m512i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u32x16 |bidirectional| core::arch::x86_64::__m512i }
*/
