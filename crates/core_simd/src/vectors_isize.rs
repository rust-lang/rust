define_type! {
    #[doc = "Vector of two `isize` values"]
    struct isizex2([isize; 2]);
}

define_type! {
    #[doc = "Vector of four `isize` values"]
    struct isizex4([isize; 4]);
}

define_type! {
    #[doc = "Vector of eight `isize` values"]
    struct isizex8([isize; 8]);
}

#[cfg(all(target_arch = "x86", target_pointer_width = "32"))]
from_aligned! { unsafe isizex4 |bidirectional| core::arch::x86::__m128i }

#[cfg(all(target_arch = "x86_64", target_pointer_width = "32"))]
from_aligned! { unsafe isizex4 |bidirectional| core::arch::x86_64::__m128i }

#[cfg(all(target_arch = "x86", target_pointer_width = "32"))]
from_aligned! { unsafe isizex8 |bidirectional| core::arch::x86::__m256i }

#[cfg(all(target_arch = "x86_64", target_pointer_width = "32"))]
from_aligned! { unsafe isizex8 |bidirectional| core::arch::x86_64::__m256i }

#[cfg(all(target_arch = "x86", target_pointer_width = "64"))]
from_aligned! { unsafe isizex2 |bidirectional| core::arch::x86::__m128i }

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
from_aligned! { unsafe isizex2 |bidirectional| core::arch::x86_64::__m128i }

#[cfg(all(target_arch = "x86", target_pointer_width = "64"))]
from_aligned! { unsafe isizex4 |bidirectional| core::arch::x86::__m256i }

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
from_aligned! { unsafe isizex4 |bidirectional| core::arch::x86_64::__m256i }

/*
#[cfg(all(target_arch = "x86", target_pointer_width = "64"))]
from_aligned! { unsafe isizex8 |bidirectional| core::arch::x86::__m512i }

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
from_aligned! { unsafe isizex8 |bidirectional| core::arch::x86_64::__m512i }
*/
