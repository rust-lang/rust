use crate::masksize;

define_mask_vector! {
    #[doc = "vector of two `masksize` values"]
    struct masksizex2([isize as masksize; 2]);
}

define_mask_vector! {
    #[doc = "vector of four `masksize` values"]
    struct masksizex4([isize as masksize; 4]);
}

define_mask_vector! {
    #[doc = "vector of eight `masksize` values"]
    struct masksizex8([isize as masksize; 8]);
}
