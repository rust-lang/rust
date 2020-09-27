define_integer_vector! {
    /// Vector of two `u32` values
    struct u32x2([u32; 2]);
}

define_integer_vector! {
    /// Vector of four `u32` values
    struct u32x4([u32; 4]);
}

define_integer_vector! {
    /// Vector of eight `u32` values
    struct u32x8([u32; 8]);
}

define_integer_vector! {
    /// Vector of 16 `u32` values
    struct u32x16([u32; 16]);
}

from_transmute_x86! { unsafe u32x4 => __m128i }
from_transmute_x86! { unsafe u32x8 => __m256i }
//from_transmute_x86! { unsafe u32x16 => __m512i }
