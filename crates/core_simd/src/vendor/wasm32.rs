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

#[cfg(target_pointer_width = "32")]
mod p32 {
    use super::*;
    from_transmute! { unsafe usizex4 => v128 }
    from_transmute! { unsafe isizex4 => v128 }
}

#[cfg(target_pointer_width = "64")]
mod p64 {
    use super::*;
    from_transmute! { unsafe usizex2 => v128 }
    from_transmute! { unsafe isizex2 => v128 }
}
