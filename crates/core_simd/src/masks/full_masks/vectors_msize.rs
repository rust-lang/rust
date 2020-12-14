use super::msize;

define_mask_vector! {
    /// Vector of two `msize` values
    struct msizex2([isize as msize; 2]);
}

define_mask_vector! {
    /// Vector of four `msize` values
    struct msizex4([isize as msize; 4]);
}

define_mask_vector! {
    /// Vector of eight `msize` values
    struct msizex8([isize as msize; 8]);
}
