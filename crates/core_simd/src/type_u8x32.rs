define_type! { struct u8x32([u8; 32]); }

#[cfg(target_arch = "x86")]
from_aligned! { unsafe u8x32 |bidirectional| core::arch::x86::__m256i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u8x32 |bidirectional| core::arch::x86_64::__m256i }
