use crate::mask8;

define_mask_vector! {
    #[doc = "vector of two `mask8` values"]
    struct mask8x2([i8 as mask8; 2]);
}

define_mask_vector! {
    #[doc = "vector of four `mask8` values"]
    struct mask8x4([i8 as mask8; 4]);
}

define_mask_vector! {
    #[doc = "vector of eight `mask8` values"]
    struct mask8x8([i8 as mask8; 8]);
}

define_mask_vector! {
    #[doc = "vector of 16 `mask8` values"]
    struct mask8x16([i8 as mask8; 16]);
}

define_mask_vector! {
    #[doc = "vector of 32 `mask8` values"]
    struct mask8x32([i8 as mask8; 32]);
}

define_mask_vector! {
    #[doc = "vector of 64 `mask8` values"]
    struct mask8x64([i8 as mask8; 64]);
}
