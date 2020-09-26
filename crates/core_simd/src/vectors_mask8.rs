use crate::mask8;

define_mask_vector! {
    /// Vector of eight `mask8` values
    struct mask8x8([i8 as mask8; 8]);
}

define_mask_vector! {
    /// Vector of 16 `mask8` values
    struct mask8x16([i8 as mask8; 16]);
}

define_mask_vector! {
    /// Vector of 32 `mask8` values
    struct mask8x32([i8 as mask8; 32]);
}

define_mask_vector! {
    /// Vector of 64 `mask8` values
    struct mask8x64([i8 as mask8; 64]);
}
