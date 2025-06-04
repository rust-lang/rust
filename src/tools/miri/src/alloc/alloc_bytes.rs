use std::alloc::Layout;
use std::borrow::Cow;
use std::{alloc, slice};
#[cfg(target_os = "linux")]
use std::{cell::RefCell, rc::Rc};

use rustc_abi::{Align, Size};
use rustc_middle::mir::interpret::AllocBytes;

#[cfg(target_os = "linux")]
use crate::alloc::isolated_alloc::IsolatedAlloc;
use crate::helpers::ToU64 as _;

#[derive(Clone, Debug)]
pub enum MiriAllocParams {
    Global,
    #[cfg(target_os = "linux")]
    Isolated(Rc<RefCell<IsolatedAlloc>>),
}

/// Allocation bytes that explicitly handle the layout of the data they're storing.
/// This is necessary to interface with native code that accesses the program store in Miri.
#[derive(Debug)]
pub struct MiriAllocBytes {
    /// Stored layout information about the allocation.
    layout: alloc::Layout,
    /// Pointer to the allocation contents.
    /// Invariant:
    /// * If `self.layout.size() == 0`, then `self.ptr` was allocated with the equivalent layout with size 1.
    /// * Otherwise, `self.ptr` points to memory allocated with `self.layout`.
    ptr: *mut u8,
    /// Whether this instance of `MiriAllocBytes` had its allocation created by calling `alloc::alloc()`
    /// (`Global`) or the discrete allocator (`Isolated`)
    params: MiriAllocParams,
}

impl Clone for MiriAllocBytes {
    fn clone(&self) -> Self {
        let bytes: Cow<'_, [u8]> = Cow::Borrowed(self);
        let align = Align::from_bytes(self.layout.align().to_u64()).unwrap();
        MiriAllocBytes::from_bytes(bytes, align, self.params.clone())
    }
}

impl Drop for MiriAllocBytes {
    fn drop(&mut self) {
        // We have to reconstruct the actual layout used for allocation.
        // (`Deref` relies on `size` so we can't just always set it to at least 1.)
        let alloc_layout = if self.layout.size() == 0 {
            Layout::from_size_align(1, self.layout.align()).unwrap()
        } else {
            self.layout
        };

        // SAFETY: Invariant, `self.ptr` points to memory allocated with `self.layout`.
        unsafe {
            match self.params.clone() {
                MiriAllocParams::Global => alloc::dealloc(self.ptr, alloc_layout),
                #[cfg(target_os = "linux")]
                MiriAllocParams::Isolated(alloc) =>
                    alloc.borrow_mut().dealloc(self.ptr, alloc_layout),
            }
        }
    }
}

impl std::ops::Deref for MiriAllocBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        // SAFETY: `ptr` is non-null, properly aligned, and valid for reading out `self.layout.size()`-many bytes.
        // Note that due to the invariant this is true even if `self.layout.size() == 0`.
        unsafe { slice::from_raw_parts(self.ptr, self.layout.size()) }
    }
}

impl std::ops::DerefMut for MiriAllocBytes {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: `ptr` is non-null, properly aligned, and valid for reading out `self.layout.size()`-many bytes.
        // Note that due to the invariant this is true even if `self.layout.size() == 0`.
        unsafe { slice::from_raw_parts_mut(self.ptr, self.layout.size()) }
    }
}

impl MiriAllocBytes {
    /// This method factors out how a `MiriAllocBytes` object is allocated, given a specific allocation function.
    /// If `size == 0` we allocate using a different `alloc_layout` with `size = 1`, to ensure each allocation has a unique address.
    /// Returns `Err(alloc_layout)` if the allocation function returns a `ptr` where `ptr.is_null()`.
    fn alloc_with(
        size: u64,
        align: u64,
        params: MiriAllocParams,
        alloc_fn: impl FnOnce(Layout, &MiriAllocParams) -> *mut u8,
    ) -> Result<MiriAllocBytes, ()> {
        let size = usize::try_from(size).map_err(|_| ())?;
        let align = usize::try_from(align).map_err(|_| ())?;
        let layout = Layout::from_size_align(size, align).map_err(|_| ())?;
        // When size is 0 we allocate 1 byte anyway, to ensure each allocation has a unique address.
        let alloc_layout =
            if size == 0 { Layout::from_size_align(1, align).unwrap() } else { layout };
        let ptr = alloc_fn(alloc_layout, &params);
        if ptr.is_null() {
            Err(())
        } else {
            // SAFETY: All `MiriAllocBytes` invariants are fulfilled.
            Ok(Self { ptr, layout, params })
        }
    }
}

impl AllocBytes for MiriAllocBytes {
    type AllocParams = MiriAllocParams;

    fn from_bytes<'a>(
        slice: impl Into<Cow<'a, [u8]>>,
        align: Align,
        params: MiriAllocParams,
    ) -> Self {
        let slice = slice.into();
        let size = slice.len();
        let align = align.bytes();
        // SAFETY: `alloc_fn` will only be used with `size != 0`.
        let alloc_fn = |layout, params: &MiriAllocParams| unsafe {
            match params {
                MiriAllocParams::Global => alloc::alloc(layout),
                #[cfg(target_os = "linux")]
                MiriAllocParams::Isolated(alloc) => alloc.borrow_mut().alloc(layout),
            }
        };
        let alloc_bytes = MiriAllocBytes::alloc_with(size.to_u64(), align, params, alloc_fn)
            .unwrap_or_else(|()| {
                panic!("Miri ran out of memory: cannot create allocation of {size} bytes")
            });
        // SAFETY: `alloc_bytes.ptr` and `slice.as_ptr()` are non-null, properly aligned
        // and valid for the `size`-many bytes to be copied.
        unsafe { alloc_bytes.ptr.copy_from(slice.as_ptr(), size) };
        alloc_bytes
    }

    fn zeroed(size: Size, align: Align, params: MiriAllocParams) -> Option<Self> {
        let size = size.bytes();
        let align = align.bytes();
        // SAFETY: `alloc_fn` will only be used with `size != 0`.
        let alloc_fn = |layout, params: &MiriAllocParams| unsafe {
            match params {
                MiriAllocParams::Global => alloc::alloc_zeroed(layout),
                #[cfg(target_os = "linux")]
                MiriAllocParams::Isolated(alloc) => alloc.borrow_mut().alloc_zeroed(layout),
            }
        };
        MiriAllocBytes::alloc_with(size, align, params, alloc_fn).ok()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
}
