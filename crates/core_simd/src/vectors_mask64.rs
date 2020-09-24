use crate::mask64;

define_mask_vector! {
    #[doc = "vector of two `mask64` values"]
    struct mask64x2([i64 as mask64; 2]);
}

define_mask_vector! {
    #[doc = "vector of four `mask64` values"]
    struct mask64x4([i64 as mask64; 4]);
}

define_mask_vector! {
    #[doc = "vector of eight `mask64` values"]
    struct mask64x8([i64 as mask64; 8]);
}
