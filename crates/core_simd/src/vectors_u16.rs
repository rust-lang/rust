define_vector! {
    #[doc = "Vector of two `u16` values"]
    struct u16x2([u16; 2]);
}

define_vector! {
    #[doc = "Vector of four `u16` values"]
    struct u16x4([u16; 4]);
}

define_vector! {
    #[doc = "Vector of eight `u16` values"]
    struct u16x8([u16; 8]);
}

define_vector! {
    #[doc = "Vector of 16 `u16` values"]
    struct u16x16([u16; 16]);
}

define_vector! {
    #[doc = "Vector of 32 `u16` values"]
    struct u16x32([u16; 32]);
}

from_transmute_x86! { unsafe u16x8 => __m128i }
from_transmute_x86! { unsafe u16x16 => __m256i }
//from_transmute_x86! { unsafe u16x32 => __m512i }
