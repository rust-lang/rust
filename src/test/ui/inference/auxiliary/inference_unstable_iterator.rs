#![feature(staged_api)]

#![stable(feature = "ipu_iterator", since = "1.0.0")]

#[stable(feature = "ipu_iterator", since = "1.0.0")]
pub trait IpuIterator {
    #[unstable(feature = "ipu_flatten", issue = "99999")]
    fn ipu_flatten(&self) -> u32 {
        0
    }
    #[unstable(feature = "assoc_const_ipu_iter", issue = "99999")]
    const C: i32;
}

#[stable(feature = "ipu_iterator", since = "1.0.0")]
impl IpuIterator for char {
    const C: i32 = 42;
}
