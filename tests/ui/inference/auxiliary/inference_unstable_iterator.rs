#![feature(staged_api)]
#![feature(arbitrary_self_types_pointers)]

#![stable(feature = "ipu_iterator", since = "1.0.0")]

#[stable(feature = "ipu_iterator", since = "1.0.0")]
pub trait IpuIterator {
    #[unstable(feature = "ipu_flatten", issue = "99999")]
    fn ipu_flatten(&self) -> u32 {
        0
    }

    #[unstable(feature = "ipu_flatten", issue = "99999")]
    fn ipu_by_value_vs_by_ref(self) -> u32 where Self: Sized {
        0
    }

    #[unstable(feature = "ipu_flatten", issue = "99999")]
    fn ipu_by_ref_vs_by_ref_mut(&self) -> u32 {
        0
    }

    #[unstable(feature = "ipu_flatten", issue = "99999")]
    fn ipu_by_mut_ptr_vs_by_const_ptr(self: *mut Self) -> u32 {
        0
    }

    #[unstable(feature = "assoc_const_ipu_iter", issue = "99999")]
    const C: i32;
}

#[stable(feature = "ipu_iterator", since = "1.0.0")]
impl IpuIterator for char {
    const C: i32 = 42;
}
