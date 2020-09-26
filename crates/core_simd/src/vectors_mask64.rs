use crate::mask64;

define_mask_vector! {
    /// Vector of two `mask64` values
    struct mask64x2([i64 as mask64; 2]);
}

define_mask_vector! {
    /// Vector of four `mask64` values
    struct mask64x4([i64 as mask64; 4]);
}

define_mask_vector! {
    /// Vector of eight `mask64` values
    struct mask64x8([i64 as mask64; 8]);
}
