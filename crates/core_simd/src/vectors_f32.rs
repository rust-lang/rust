define_float_vector! {
    /// Vector of two `f32` values
    struct f32x2([f32; 2]);
    bits crate::u32x2;
}

define_float_vector! {
    /// Vector of four `f32` values
    struct f32x4([f32; 4]);
    bits crate::u32x4;
}

define_float_vector! {
    /// Vector of eight `f32` values
    struct f32x8([f32; 8]);
    bits crate::u32x8;
}

define_float_vector! {
    /// Vector of 16 `f32` values
    struct f32x16([f32; 16]);
    bits crate::u32x16;
}

from_transmute_x86! { unsafe f32x4 => __m128 }
from_transmute_x86! { unsafe f32x8 => __m256 }
//from_transmute_x86! { unsafe f32x16 => __m512 }


