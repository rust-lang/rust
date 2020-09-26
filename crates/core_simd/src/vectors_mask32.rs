use crate::mask32;

define_mask_vector! {
    /// Vector of two `mask32` values
    struct mask32x2([i32 as mask32; 2]);
}

define_mask_vector! {
    /// Vector of four `mask32` values
    struct mask32x4([i32 as mask32; 4]);
}

define_mask_vector! {
    /// Vector of eight `mask32` values
    struct mask32x8([i32 as mask32; 8]);
}

define_mask_vector! {
    /// Vector of 16 `mask32` values
    struct mask32x16([i32 as mask32; 16]);
}
