define_integer_vector! {
    /// Vector of two `i64` values
    struct i64x2([i64; 2]);
}

define_integer_vector! {
    /// Vector of four `i64` values
    struct i64x4([i64; 4]);
}

define_integer_vector! {
    /// Vector of eight `i64` values
    struct i64x8([i64; 8]);
}

from_transmute_x86! { unsafe i64x2 => __m128i }
from_transmute_x86! { unsafe i64x4 => __m256i }
//from_transmute_x86! { unsafe i64x8 => __m512i }
