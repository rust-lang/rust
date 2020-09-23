define_type! {
    #[doc = "Vector of eight `usize` types"]
    struct usizex8([usize; 8]);
}

#[cfg(all(target_arch = "x86", target_pointer_width = "32"))]
from_aligned! { unsafe usizex8 |bidirectional| core::arch::x86::__m256i }

#[cfg(all(target_arch = "x86_64", target_pointer_width = "32"))]
from_aligned! { unsafe usizex8 |bidirectional| core::arch::x86_64::__m256i }

/*
#[cfg(all(target_arch = "x86", target_pointer_width = "64"))]
from_aligned! { unsafe usizex8 |bidirectional| core::arch::x86::__m512i }

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
from_aligned! { unsafe usizex8 |bidirectional| core::arch::x86_64::__m512i }
*/
