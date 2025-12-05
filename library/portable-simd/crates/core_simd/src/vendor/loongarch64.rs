use crate::simd::*;
use core::arch::loongarch64::*;

from_transmute! { unsafe u8x16 => v16u8 }
from_transmute! { unsafe u8x32 => v32u8 }
from_transmute! { unsafe i8x16 => v16i8 }
from_transmute! { unsafe i8x32 => v32i8 }

from_transmute! { unsafe u16x8 => v8u16 }
from_transmute! { unsafe u16x16 => v16u16 }
from_transmute! { unsafe i16x8 => v8i16 }
from_transmute! { unsafe i16x16 => v16i16 }

from_transmute! { unsafe u32x4 => v4u32 }
from_transmute! { unsafe u32x8 => v8u32 }
from_transmute! { unsafe i32x4 => v4i32 }
from_transmute! { unsafe i32x8 => v8i32 }
from_transmute! { unsafe f32x4 => v4f32 }
from_transmute! { unsafe f32x8 => v8f32 }

from_transmute! { unsafe u64x2 => v2u64 }
from_transmute! { unsafe u64x4 => v4u64 }
from_transmute! { unsafe i64x2 => v2i64 }
from_transmute! { unsafe i64x4 => v4i64 }
from_transmute! { unsafe f64x2 => v2f64 }
from_transmute! { unsafe f64x4 => v4f64 }

from_transmute! { unsafe usizex2 => v2u64 }
from_transmute! { unsafe usizex4 => v4u64 }
from_transmute! { unsafe isizex2 => v2i64 }
from_transmute! { unsafe isizex4 => v4i64 }
