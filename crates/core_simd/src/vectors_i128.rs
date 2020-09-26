define_vector! {
    /// Vector of two `i128` values
    #[derive(Eq, Ord, Hash)]
    struct i128x2([i128; 2]);
}

define_vector! {
    /// Vector of four `i128` values
    #[derive(Eq, Ord, Hash)]
    struct i128x4([i128; 4]);
}

from_transmute_x86! { unsafe i128x2 => __m256i }
//from_transmute_x86! { unsafe i128x4 => __m512i }
