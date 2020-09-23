define_type! {
    #[doc = "Vector of two `i32` values"]
    struct i32x2([i32; 2]);
}

define_type! {
    #[doc = "Vector of four `i32` values"]
    struct i32x4([i32; 4]);
}

define_type! {
    #[doc = "Vector of eight `i32` values"]
    struct i32x8([i32; 8]);
}

define_type! {
    #[doc = "Vector of 16 `i32` values"]
    struct i32x16([i32; 16]);
}

from_transmute_x86! { unsafe i32x4 => __m128i }
from_transmute_x86! { unsafe i32x8 => __m256i }
//from_transmute_x86! { unsafe i32x16 => __m512i }
