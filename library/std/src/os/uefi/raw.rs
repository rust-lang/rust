//! This module just re-exports stuff from r-efi crate

pub use r_efi::efi::{BootServices, Guid, RuntimeServices, SystemTable};

use crate::alloc::{Allocator, Global, Layout};
use crate::io;
use crate::ptr::NonNull;
use crate::sys_common::AsInner;

// A type to make working with Variable Sized Types easier
pub(crate) struct VariableSizeType<T> {
    inner: NonNull<T>,
    layout: Layout,
}

impl<T> VariableSizeType<T> {
    const ALIGNMENT: usize = 8;

    pub(crate) fn new(inner: NonNull<T>, layout: Layout) -> Self {
        Self { inner, layout }
    }

    pub(crate) fn from_size(size: usize) -> io::Result<Self> {
        let layout = Layout::from_size_align(size, Self::ALIGNMENT).map_err(|_| {
            io::error::const_io_error!(io::ErrorKind::Uncategorized, "Invalid buffer size")
        })?;
        let inner: NonNull<T> = Global
            .allocate(layout)
            .map_err(|_| {
                io::error::const_io_error!(
                    io::ErrorKind::Uncategorized,
                    "Failed to allocate Buffer"
                )
            })?
            .cast();
        Ok(Self::new(inner, layout))
    }

    pub(crate) fn as_ptr(&self) -> *mut T {
        self.inner.as_ptr()
    }

    // Callers responsibility to ensure that it has been inintialized beforehand
    pub(crate) fn as_ref(&self) -> &T {
        unsafe { self.inner.as_ref() }
    }

    pub(crate) fn size(&self) -> usize {
        self.layout.size()
    }
}

impl<T> AsInner<NonNull<T>> for VariableSizeType<T> {
    fn as_inner(&self) -> &NonNull<T> {
        &self.inner
    }
}

impl<T> Drop for VariableSizeType<T> {
    fn drop(&mut self) {
        unsafe { Global.deallocate(self.inner.cast(), self.layout) }
    }
}
