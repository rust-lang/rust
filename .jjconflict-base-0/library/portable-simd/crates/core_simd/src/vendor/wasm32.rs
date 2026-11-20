use crate::simd::*;
use core::arch::wasm32::v128;

from_transmute! { unsafe u8x16 => v128 }
from_transmute! { unsafe i8x16 => v128 }

from_transmute! { unsafe u16x8 => v128 }
from_transmute! { unsafe i16x8 => v128 }

from_transmute! { unsafe u32x4 => v128 }
from_transmute! { unsafe i32x4 => v128 }
from_transmute! { unsafe f32x4 => v128 }

from_transmute! { unsafe u64x2 => v128 }
from_transmute! { unsafe i64x2 => v128 }
from_transmute! { unsafe f64x2 => v128 }
