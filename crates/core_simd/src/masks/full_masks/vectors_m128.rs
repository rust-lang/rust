use super::m128;

define_mask_vector! {
    /// Vector of two `m128` values
    struct m128x2([i128 as m128; 2]);
}

define_mask_vector! {
    /// Vector of four `m128` values
    struct m128x4([i128 as m128; 4]);
}
