define_type! {
    #[doc = "Vector of two `usize` types"]
    struct usizex2([usize; 2]);
}

#[cfg(all(target_arch = "x86", target_pointer_width = "64"))]
from_aligned! { unsafe usizex2 |bidirectional| core::arch::x86::__m128i }

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
from_aligned! { unsafe usizex2 |bidirectional| core::arch::x86_64::__m128i }
