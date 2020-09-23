define_type! {
    #[doc = "Vector of two `usize` values"]
    struct usizex2([usize; 2]);
}

define_type! {
    #[doc = "Vector of four `usize` values"]
    struct usizex4([usize; 4]);
}

define_type! {
    #[doc = "Vector of eight `usize` values"]
    struct usizex8([usize; 8]);
}

#[cfg(all(target_arch = "x86", target_pointer_width = "32"))]
from_aligned! { unsafe usizex4 |bidirectional| core::arch::x86::__m128i }

#[cfg(all(target_arch = "x86_64", target_pointer_width = "32"))]
from_aligned! { unsafe usizex4 |bidirectional| core::arch::x86_64::__m128i }

#[cfg(all(target_arch = "x86", target_pointer_width = "32"))]
from_aligned! { unsafe usizex8 |bidirectional| core::arch::x86::__m256i }

#[cfg(all(target_arch = "x86_64", target_pointer_width = "32"))]
from_aligned! { unsafe usizex8 |bidirectional| core::arch::x86_64::__m256i }

#[cfg(all(target_arch = "x86", target_pointer_width = "64"))]
from_aligned! { unsafe usizex2 |bidirectional| core::arch::x86::__m128i }

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
from_aligned! { unsafe usizex2 |bidirectional| core::arch::x86_64::__m128i }

#[cfg(all(target_arch = "x86", target_pointer_width = "64"))]
from_aligned! { unsafe usizex4 |bidirectional| core::arch::x86::__m256i }

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
from_aligned! { unsafe usizex4 |bidirectional| core::arch::x86_64::__m256i }

/*
#[cfg(all(target_arch = "x86", target_pointer_width = "64"))]
from_aligned! { unsafe usizex8 |bidirectional| core::arch::x86::__m512i }

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
from_aligned! { unsafe usizex8 |bidirectional| core::arch::x86_64::__m512i }
*/
