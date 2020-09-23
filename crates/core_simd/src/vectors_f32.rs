define_type! {
    #[doc = "Vector of two `f32` values"]
    struct f32x2([f32; 2]);
}

define_type! {
    #[doc = "Vector of four `f32` values"]
    struct f32x4([f32; 4]);
}

define_type! {
    #[doc = "Vector of eight `f32` values"]
    struct f32x8([f32; 8]);
}

define_type! {
    #[doc = "Vector of 16 `f32` values"]
    struct f32x16([f32; 16]);
}

from_transmute_x86! { unsafe f32x4 => __m128 }
from_transmute_x86! { unsafe f32x8 => __m256 }
//from_transmute_x86! { unsafe f32x16 => __m512 }
