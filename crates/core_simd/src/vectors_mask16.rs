use crate::mask16;

define_mask_vector! {
    #[doc = "vector of two `mask16` values"]
    struct mask16x2([i16 as mask16; 2]);
}

define_mask_vector! {
    #[doc = "vector of four `mask16` values"]
    struct mask16x4([i16 as mask16; 4]);
}

define_mask_vector! {
    #[doc = "vector of eight `mask16` values"]
    struct mask16x8([i16 as mask16; 8]);
}

define_mask_vector! {
    #[doc = "vector of 16 `mask16` values"]
    struct mask16x16([i16 as mask16; 16]);
}

define_mask_vector! {
    #[doc = "vector of 32 `mask16` values"]
    struct mask16x32([i16 as mask16; 32]);
}
