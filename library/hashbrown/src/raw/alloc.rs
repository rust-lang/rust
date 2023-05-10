pub(crate) use self::inner::{do_alloc, Allocator, Global};

#[cfg(feature = "nightly")]
mod inner {
    use crate::alloc::alloc::Layout;
    pub use crate::alloc::alloc::{Allocator, Global};
    use core::ptr::NonNull;

    #[allow(clippy::map_err_ignore)]
    pub(crate) fn do_alloc<A: Allocator>(alloc: &A, layout: Layout) -> Result<NonNull<u8>, ()> {
        match alloc.allocate(layout) {
            Ok(ptr) => Ok(ptr.as_non_null_ptr()),
            Err(_) => Err(()),
        }
    }
}

#[cfg(not(feature = "nightly"))]
mod inner {
    use crate::alloc::alloc::Layout;
    pub use allocator_api2::alloc::{Allocator, Global};
    use core::ptr::NonNull;

    #[allow(clippy::map_err_ignore)]
    pub(crate) fn do_alloc<A: Allocator>(alloc: &A, layout: Layout) -> Result<NonNull<u8>, ()> {
        match alloc.allocate(layout) {
            Ok(ptr) => Ok(ptr.cast()),
            Err(_) => Err(()),
        }
    }
}
