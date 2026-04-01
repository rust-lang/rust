use crate::simd::*;
use core::arch::loongarch64::*;

from_transmute! { unsafe u8x16 => m128i }
from_transmute! { unsafe u8x32 => m256i }
from_transmute! { unsafe i8x16 => m128i }
from_transmute! { unsafe i8x32 => m256i }

from_transmute! { unsafe u16x8 => m128i }
from_transmute! { unsafe u16x16 => m256i }
from_transmute! { unsafe i16x8 => m128i }
from_transmute! { unsafe i16x16 => m256i }

from_transmute! { unsafe u32x4 => m128i }
from_transmute! { unsafe u32x8 => m256i }
from_transmute! { unsafe i32x4 => m128i }
from_transmute! { unsafe i32x8 => m256i }
from_transmute! { unsafe f32x4 => m128 }
from_transmute! { unsafe f32x8 => m256 }

from_transmute! { unsafe u64x2 => m128i }
from_transmute! { unsafe u64x4 => m256i }
from_transmute! { unsafe i64x2 => m128i }
from_transmute! { unsafe i64x4 => m256i }
from_transmute! { unsafe f64x2 => m128d }
from_transmute! { unsafe f64x4 => m256d }
