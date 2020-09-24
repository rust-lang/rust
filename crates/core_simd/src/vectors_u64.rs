define_vector! {
    #[doc = "Vector of two `u64` values"]
    struct u64x2([u64; 2]);
}

define_vector! {
    #[doc = "Vector of four `u64` values"]
    struct u64x4([u64; 4]);
}

define_vector! {
    #[doc = "Vector of eight `u64` values"]
    struct u64x8([u64; 8]);
}

from_transmute_x86! { unsafe u64x2 => __m128i }
from_transmute_x86! { unsafe u64x4 => __m256i }
//from_transmute_x86! { unsafe u64x8 => __m512i }
