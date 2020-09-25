define_vector! {
    #[doc = "Vector of two `i16` values"]
    #[derive(Eq, Ord, Hash)]
    struct i16x2([i16; 2]);
}

define_vector! {
    #[doc = "Vector of four `i16` values"]
    #[derive(Eq, Ord, Hash)]
    struct i16x4([i16; 4]);
}

define_vector! {
    #[doc = "Vector of eight `i16` values"]
    #[derive(Eq, Ord, Hash)]
    struct i16x8([i16; 8]);
}

define_vector! {
    #[doc = "Vector of 16 `i16` values"]
    #[derive(Eq, Ord, Hash)]
    struct i16x16([i16; 16]);
}

define_vector! {
    #[doc = "Vector of 32 `i16` values"]
    #[derive(Eq, Ord, Hash)]
    struct i16x32([i16; 32]);
}

from_transmute_x86! { unsafe i16x8 => __m128i }
from_transmute_x86! { unsafe i16x16 => __m256i }
//from_transmute_x86! { unsafe i16x32 => __m512i }
