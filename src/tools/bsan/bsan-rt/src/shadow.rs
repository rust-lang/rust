use core::alloc::Layout;
use core::ffi::c_void;
use core::marker::PhantomData;
use core::ops::{Add, BitAnd, Deref, DerefMut, Shr};
use core::{mem, ptr};

use libc::{MAP_ANONYMOUS, MAP_NORESERVE, MAP_PRIVATE, PROT_READ, PROT_WRITE};

use crate::BsanAllocator;
use crate::global::{GlobalContext, global_ctx};

/// Different targets have a different number
/// of significant bits in their pointer representation.
/// On 32-bit platforms, all 32-bits are addressable. Most
/// 64-bit platforms only use 48-bits. Following the LLVM Project,
/// we hard-code these values based on the underlying architecture.
/// Most, if not all 64 bit architectures use 48-bits. However, a the
/// Armv8-A spec allows addressing 52 or 56 bits as well. No processors
/// implement this yet, though, so we can use target_pointer_width.

#[cfg(target_pointer_width = "64")]
static VA_BITS: usize = 48;

#[cfg(target_pointer_width = "32")]
static VA_BITS: usize = 32;

#[cfg(target_pointer_width = "16")]
static VA_BITS: usize = 16;

// The number of bytes in a pointer
static PTR_BYTES: usize = mem::size_of::<usize>();

// The number of addressable, word-aligned, pointer-sized chunks
static NUM_ADDR_CHUNKS: u32 = VA_BITS.strict_div(PTR_BYTES).ilog2();

// We have 2^L2_POWER entries in the second level of the page table
// Adding 1 ensures that we have more second-level entries than first
// level entries if the number of addressable chunks is odd.
static L2_POWER: u32 = NUM_ADDR_CHUNKS.strict_add(1).strict_div(2);

// We have 2^L1_POWER entries in the first level of the page table
static L1_POWER: u32 = NUM_ADDR_CHUNKS.strict_div(2);

// The number of entries in the second level of the page table
static L2_LEN: usize = 2_usize.pow(L2_POWER);

// The number of entries in the first level of the page table
static L1_LEN: usize = 2_usize.pow(L1_POWER);

/// Converts an address into a pair of indices into the first and second
/// levels of the shadow page table.
#[inline(always)]
pub fn table_indices(address: usize) -> (usize, usize) {
    #[cfg(target_endian = "little")]
    let l1_index = address.shr(L2_POWER).bitand((L1_POWER - 1) as usize);

    #[cfg(target_endian = "big")]
    let l1_index = address.shl(L2_POWER).bitand((L1_POWER - 1) as usize);

    let l2_index = address.bitand((L2_POWER - 1) as usize);

    (l1_index, l2_index)
}

// Provenance values must be sized so that we can allocate an array of them
// for the L1 page table. We can make provenance values Copy since they should
// fit within 128 bits and they are not "owned" by any particular object.
pub trait Provenance: Copy + Sized {}

#[repr(C)]
pub struct L2<T: Provenance> {
    pub bytes: *mut [T; L2_LEN],
}

impl<T: Provenance> L2<T> {
    pub fn new(allocator: BsanAllocator) -> Self {
        let mut l2_bytes: *mut [T; L2_LEN] = unsafe {
            let l2_void = (allocator.mmap)(
                core::ptr::null_mut(),
                size_of::<T>() * L2_LEN,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                -1,
                0,
            );

            ptr::write_bytes(l2_void as *mut u8, 0, size_of::<T>() * L2_LEN);
            mem::transmute(l2_void)
        };

        Self { bytes: l2_bytes }
    }
    #[inline(always)]
    pub unsafe fn lookup_mut(&mut self, index: usize) -> &mut T {
        self.bytes.get_unchecked_mut(index)
    }
    #[inline(always)]
    pub unsafe fn lookup(&mut self, index: usize) -> &T {
        self.bytes.get_unchecked(index)
    }
}

#[repr(C)]
pub struct L1<T: Provenance> {
    pub entries: *mut [*mut L2<T>; L1_LEN], //
}

impl<T: Provenance> L1<T> {
    pub fn new(allocator: BsanAllocator) -> Self {
        let mut l1_entries: *mut [*mut L2<T>; L1_LEN] = unsafe {
            let l1_void = (allocator.mmap)(
                core::ptr::null_mut(),
                PTR_BYTES * L1_LEN,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                -1,
                0,
            );
            assert!(l1_void != core::ptr::null_mut() || l1_void != -1isize as (*mut c_void));
            // zero bytes after allocating
            ptr::write_bytes(l1_void as *mut u8, 0, PTR_BYTES * L1_LEN);
            mem::transmute(l1_void)
        };

        Self { entries: l1_entries }
    }

    #[inline(always)]
    unsafe fn lookup_mut(&mut self, index: usize) -> Option<&mut T> {
        let (l1_index, l2_index) = table_indices(index);
        let l2 = self.entries.get_unchecked_mut(l1_index);
        if l2.is_null() { None } else { Some((**l2).lookup_mut(l2_index)) }
    }

    #[inline(always)]
    unsafe fn lookup(&mut self, index: usize) -> Option<&T> {
        let (l1_index, l2_index) = table_indices(index);
        let l2 = self.entries.get_unchecked(l1_index);
        if l2.is_null() { None } else { Some((**l2).lookup(l2_index)) }
    }
}

/// A two-level page table. This wrapper struct encapsulates
/// the interior, unsafe implementation, providing debug assertions
/// for each method.
#[repr(transparent)]
#[derive(Debug)]
pub struct ShadowHeap<T: Provenance> {
    pub(crate) l1: L1<T>,
}

impl<T: Provenance> Default for ShadowHeap<T> {
    fn default() -> Self {
        let l1 = unsafe {
            let allocator = global_ctx().allocator;
            L1::new(allocator)
        };
        Self { l1 }
    }
}

impl<T: Provenance> Deref for ShadowHeap<T> {
    type Target = L1<T>;
    fn deref(&self) -> &Self::Target {
        &self.l1
    }
}

impl<T: Provenance> DerefMut for ShadowHeap<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.l1
    }
}

impl ShadowHeap<T: Provenance> {
    unsafe fn malloc(&mut self, ptr: *mut u8, size: usize) {
        if size == 0 {
            return;
        }
        let (l1_addr, l2_addr) = table_indices(ptr);
        match self.l1.lookup_mut(l1_addr) {
            None => {
                // if there is no valid L2, it will initialize
            }
            Some(l2) => {}
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type TestProv = u8;

    impl Provenance for TestProv {}

    #[test]
    fn create_and_drop() {
        let _ = ShadowHeap::<TestProv>::default();
    }
}
