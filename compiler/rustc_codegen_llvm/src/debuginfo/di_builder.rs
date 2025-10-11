use std::ptr;

use crate::llvm::debuginfo::DIBuilder;
use crate::llvm::{self, Module};

/// Owning pointer to a `DIBuilder<'ll>` that will dispose of the builder
/// when dropped. Use `.as_ref()` to get the underlying `&DIBuilder`
/// needed for debuginfo FFI calls.
pub(crate) struct DIBuilderBox<'ll> {
    raw: ptr::NonNull<DIBuilder<'ll>>,
}

impl<'ll> DIBuilderBox<'ll> {
    pub(crate) fn new(llmod: &'ll Module) -> Self {
        let raw = unsafe { llvm::LLVMCreateDIBuilder(llmod) };
        let raw = ptr::NonNull::new(raw).unwrap();
        Self { raw }
    }

    pub(crate) fn as_ref(&self) -> &DIBuilder<'ll> {
        // SAFETY: This is an owning pointer, so `&DIBuilder` is valid
        // for as long as `&self` is.
        unsafe { self.raw.as_ref() }
    }
}

impl<'ll> Drop for DIBuilderBox<'ll> {
    fn drop(&mut self) {
        unsafe { llvm::LLVMDisposeDIBuilder(self.raw) };
    }
}
