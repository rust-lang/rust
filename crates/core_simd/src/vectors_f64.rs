define_type! {
    #[doc = "Vector of two `f64` values"]
    struct f64x2([f64; 2]);
}

define_type! {
    #[doc = "Vector of four `f64` values"]
    struct f64x4([f64; 4]);
}

define_type! {
    #[doc = "Vector of eight `f64` values"]
    struct f64x8([f64; 8]);
}

from_transmute_x86! { unsafe f64x2 => __m128d }
from_transmute_x86! { unsafe f64x4 => __m256d }
//from_transmute_x86! { unsafe f64x8 => __m512d }
