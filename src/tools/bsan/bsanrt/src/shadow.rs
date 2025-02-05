use core::alloc::Layout;
use core::marker::PhantomData;
use core::mem;
use core::ops::{Add, BitAnd, Deref, DerefMut, Shr};
use alloc::vec::Vec;
use alloc::alloc::{alloc, dealloc};

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
fn table_indices(address: usize) -> (usize, usize) {
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
    bytes: [T; L2_LEN],
}

impl<T: Provenance> L2<T> {
    #[inline(always)]
    unsafe fn lookup_mut(&mut self, index: usize) -> &mut T {
        self.bytes.get_unchecked_mut(index)
    }
    #[inline(always)]
    unsafe fn lookup(&mut self, index: usize) -> &T {
        self.bytes.get_unchecked(index)
    }
}

#[repr(C)]
pub struct L1<T: Provenance> {
    entries: [*mut L2<T>; L1_LEN],
    // We need to keep track of all the chunks that we allocate
    // so that we can deallocate them when the shadow heap is dropped.
    chunks: Vec<*mut L2<T>>,
}

impl<T: Provenance> L1<T> {
    fn new() -> Self {
        Self { entries: [core::ptr::null_mut(); L1_LEN], chunks: Vec::new() }
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

impl<T: Provenance> Drop for L1<T> {
    fn drop(&mut self) {
        for chunk in self.chunks.iter() {
            unsafe {
                dealloc(*chunk.cast::<*mut u8>(), Layout::new::<L2<T>>());
            }
        }
    }
}

/// A two-level page table. This wrapper struct encapsulates
/// the interior, unsafe implementation, providing debug assertions
/// for each method.
#[repr(transparent)]
pub struct ShadowHeap<T: Provenance> {
    l1: L1<T>,
}

impl<T: Provenance> Default for ShadowHeap<T> {
    fn default() -> Self {
        let l1 = L1::<T>::new();
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
