define_type! { struct u32x4([u32; 4]); }

#[cfg(target_arch = "x86")]
from_aligned! { unsafe u32x4 |bidirectional| core::arch::x86::__m128i }

#[cfg(target_arch = "x86_64")]
from_aligned! { unsafe u32x4 |bidirectional| core::arch::x86_64::__m128i }
