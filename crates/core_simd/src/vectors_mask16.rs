use crate::mask16;

define_mask_vector! {
    /// Vector of four `mask16` values
    struct mask16x4([i16 as mask16; 4]);
}

define_mask_vector! {
    /// Vector of eight `mask16` values
    struct mask16x8([i16 as mask16; 8]);
}

define_mask_vector! {
    /// Vector of 16 `mask16` values
    struct mask16x16([i16 as mask16; 16]);
}

define_mask_vector! {
    /// Vector of 32 `mask16` values
    struct mask16x32([i16 as mask16; 32]);
}
