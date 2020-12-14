use super::m64;

define_mask_vector! {
    /// Vector of two `m64` values
    struct m64x2([i64 as m64; 2]);
}

define_mask_vector! {
    /// Vector of four `m64` values
    struct m64x4([i64 as m64; 4]);
}

define_mask_vector! {
    /// Vector of eight `m64` values
    struct m64x8([i64 as m64; 8]);
}
