define_vector! {
    #[doc = "Vector of two `i8` values"]
    #[derive(Eq, Ord, Hash)]
    struct i8x2([i8; 2]);
}

define_vector! {
    #[doc = "Vector of four `i8` values"]
    #[derive(Eq, Ord, Hash)]
    struct i8x4([i8; 4]);
}

define_vector! {
    #[doc = "Vector of eight `i8` values"]
    #[derive(Eq, Ord, Hash)]
    struct i8x8([i8; 8]);
}

define_vector! {
    #[doc = "Vector of 16 `i8` values"]
    #[derive(Eq, Ord, Hash)]
    struct i8x16([i8; 16]);
}

define_vector! {
    #[doc = "Vector of 32 `i8` values"]
    #[derive(Eq, Ord, Hash)]
    struct i8x32([i8; 32]);
}

define_vector! {
    #[doc = "Vector of 64 `i8` values"]
    #[derive(Eq, Ord, Hash)]
    struct i8x64([i8; 64]);
}

from_transmute_x86! { unsafe i8x16 => __m128i }
from_transmute_x86! { unsafe i8x32 => __m256i }
//from_transmute_x86! { unsafe i8x64 => __m512i }
