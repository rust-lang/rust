define_type! {
    #[doc = "Vector of four `usize` types"]
    struct usizex4([usize; 4]);
}

#[cfg(all(target_arch = "x86", target_pointer_width = "32"))]
from_aligned! { unsafe usizex4 |bidirectional| core::arch::x86::__m128i }

#[cfg(all(target_arch = "x86_64", target_pointer_width = "32"))]
from_aligned! { unsafe usizex4 |bidirectional| core::arch::x86_64::__m128i }

#[cfg(all(target_arch = "x86", target_pointer_width = "64"))]
from_aligned! { unsafe usizex4 |bidirectional| core::arch::x86::__m256i }

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
from_aligned! { unsafe usizex4 |bidirectional| core::arch::x86_64::__m256i }
