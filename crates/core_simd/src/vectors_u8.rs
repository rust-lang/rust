define_vector! {
    #[doc = "Vector of two `u8` values"]
    #[derive(Eq, Ord, Hash)]
    struct u8x2([u8; 2]);
}

define_vector! {
    #[doc = "Vector of four `u8` values"]
    #[derive(Eq, Ord, Hash)]
    struct u8x4([u8; 4]);
}

define_vector! {
    #[doc = "Vector of eight `u8` values"]
    #[derive(Eq, Ord, Hash)]
    struct u8x8([u8; 8]);
}

define_vector! {
    #[doc = "Vector of 16 `u8` values"]
    #[derive(Eq, Ord, Hash)]
    struct u8x16([u8; 16]);
}

define_vector! {
    #[doc = "Vector of 32 `u8` values"]
    #[derive(Eq, Ord, Hash)]
    struct u8x32([u8; 32]);
}

define_vector! {
    #[doc = "Vector of 64 `u8` values"]
    #[derive(Eq, Ord, Hash)]
    struct u8x64([u8; 64]);
}

from_transmute_x86! { unsafe u8x16 => __m128i }
from_transmute_x86! { unsafe u8x32 => __m256i }
//from_transmute_x86! { unsafe u8x64 => __m512i }
