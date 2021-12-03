use crate::simd::*;

#[cfg(any(target_arch = "x86"))]
use core::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

from_transmute! { unsafe u8x16 => __m128i }
from_transmute! { unsafe u8x32 => __m256i }
from_transmute! { unsafe u8x64 => __m512i }
from_transmute! { unsafe i8x16 => __m128i }
from_transmute! { unsafe i8x32 => __m256i }
from_transmute! { unsafe i8x64 => __m512i }

from_transmute! { unsafe u16x8 => __m128i }
from_transmute! { unsafe u16x16 => __m256i }
from_transmute! { unsafe u16x32 => __m512i }
from_transmute! { unsafe i16x8 => __m128i }
from_transmute! { unsafe i16x16 => __m256i }
from_transmute! { unsafe i16x32 => __m512i }

from_transmute! { unsafe u32x4 => __m128i }
from_transmute! { unsafe u32x8 => __m256i }
from_transmute! { unsafe u32x16 => __m512i }
from_transmute! { unsafe i32x4 => __m128i }
from_transmute! { unsafe i32x8 => __m256i }
from_transmute! { unsafe i32x16 => __m512i }
from_transmute! { unsafe f32x4 => __m128 }
from_transmute! { unsafe f32x8 => __m256 }
from_transmute! { unsafe f32x16 => __m512 }

from_transmute! { unsafe u64x2 => __m128i }
from_transmute! { unsafe u64x4 => __m256i }
from_transmute! { unsafe u64x8 => __m512i }
from_transmute! { unsafe i64x2 => __m128i }
from_transmute! { unsafe i64x4 => __m256i }
from_transmute! { unsafe i64x8 => __m512i }
from_transmute! { unsafe f64x2 => __m128d }
from_transmute! { unsafe f64x4 => __m256d }
from_transmute! { unsafe f64x8 => __m512d }

#[cfg(target_pointer_width = "32")]
mod p32 {
    use super::*;
    from_transmute! { unsafe usizex4 => __m128i }
    from_transmute! { unsafe usizex8 => __m256i }
    from_transmute! { unsafe Simd<usize, 16> => __m512i }
    from_transmute! { unsafe isizex4 => __m128i }
    from_transmute! { unsafe isizex8 => __m256i }
    from_transmute! { unsafe Simd<isize, 16> => __m512i }
}

#[cfg(target_pointer_width = "64")]
mod p64 {
    use super::*;
    from_transmute! { unsafe usizex2 => __m128i }
    from_transmute! { unsafe usizex4 => __m256i }
    from_transmute! { unsafe usizex8 => __m512i }
    from_transmute! { unsafe isizex2 => __m128i }
    from_transmute! { unsafe isizex4 => __m256i }
    from_transmute! { unsafe isizex8 => __m512i }
}
