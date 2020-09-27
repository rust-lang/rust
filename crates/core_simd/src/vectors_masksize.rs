use crate::masksize;

define_mask_vector! {
    /// Vector of two `masksize` values
    struct masksizex2([isize as masksize; 2]);
}

define_mask_vector! {
    /// Vector of four `masksize` values
    struct masksizex4([isize as masksize; 4]);
}

define_mask_vector! {
    /// Vector of eight `masksize` values
    struct masksizex8([isize as masksize; 8]);
}
