use super::m32;

define_mask_vector! {
    /// Vector of two `m32` values
    struct m32x2([i32 as m32; 2]);
}

define_mask_vector! {
    /// Vector of four `m32` values
    struct m32x4([i32 as m32; 4]);
}

define_mask_vector! {
    /// Vector of eight `m32` values
    struct m32x8([i32 as m32; 8]);
}

define_mask_vector! {
    /// Vector of 16 `m32` values
    struct m32x16([i32 as m32; 16]);
}
