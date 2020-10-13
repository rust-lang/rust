use super::m16;

define_mask_vector! {
    /// Vector of four `m16` values
    struct m16x4([i16 as m16; 4]);
}

define_mask_vector! {
    /// Vector of eight `m16` values
    struct m16x8([i16 as m16; 8]);
}

define_mask_vector! {
    /// Vector of 16 `m16` values
    struct m16x16([i16 as m16; 16]);
}

define_mask_vector! {
    /// Vector of 32 `m16` values
    struct m16x32([i16 as m16; 32]);
}
