define_type! { struct isizex4([isize; 4]); }

#[cfg(all(target_arch = "x86", target_pointer_width = "32"))]
from_aligned! { unsafe isizex4 |bidirectional| core::arch::x86::__m128i }

#[cfg(all(target_arch = "x86_64", target_pointer_width = "32"))]
from_aligned! { unsafe isizex4 |bidirectional| core::arch::x86_64::__m128i }

#[cfg(all(target_arch = "x86", target_pointer_width = "64"))]
from_aligned! { unsafe isizex4 |bidirectional| core::arch::x86::__m256i }

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
from_aligned! { unsafe isizex4 |bidirectional| core::arch::x86_64::__m256i }
