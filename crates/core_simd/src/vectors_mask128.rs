use crate::mask128;

define_mask_vector! {
    /// Vector of two `mask128` values
    struct mask128x2([i128 as mask128; 2]);
}

define_mask_vector! {
    /// Vector of four `mask128` values
    struct mask128x4([i128 as mask128; 4]);
}
