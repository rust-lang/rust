use super::m8;

define_mask_vector! {
    /// Vector of eight `m8` values
    struct m8x8([i8 as m8; 8]);
}

define_mask_vector! {
    /// Vector of 16 `m8` values
    struct m8x16([i8 as m8; 16]);
}

define_mask_vector! {
    /// Vector of 32 `m8` values
    struct m8x32([i8 as m8; 32]);
}

define_mask_vector! {
    /// Vector of 64 `m8` values
    struct m8x64([i8 as m8; 64]);
}
