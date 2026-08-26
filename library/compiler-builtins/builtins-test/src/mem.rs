extern crate alloc;

use alloc::boxed::Box;
use alloc::vec;
use core::{ops, slice};

/// 4 kiB
pub const PAGE_SIZE: usize = 0x1000;
/// 1 MiB
pub const MEG1: usize = 1 << 20;
/// When we want to test behavior that may depend on aligned reads/writes, use this value. Large
/// enough for AVX512.
pub const MAX_TESTED_ALIGN: usize = 512;

#[derive(Clone)]
#[repr(C, align(0x1000))]
struct Page([u8; PAGE_SIZE]);

/// A buffer that is page-aligned by default and dereferences to a slice, with an optional offset
/// for the deref to create a misaligned buffer.
pub struct AlignedSlice {
    buf: Box<[Page]>,
    len: usize,
    offset: usize,
}

impl AlignedSlice {
    /// Allocate a slice aligned to ALIGN with at least `len` items, with `offset` from
    /// page alignment.
    pub fn new_zeroed(len: usize, offset: usize) -> Self {
        assert!(offset < PAGE_SIZE);
        let total_len = len + offset;
        let limbs = total_len.div_ceil(PAGE_SIZE);
        let buf = vec![Page([0u8; PAGE_SIZE]); limbs].into_boxed_slice();
        AlignedSlice { buf, len, offset }
    }
}

impl ops::Deref for AlignedSlice {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.buf.as_ptr().cast::<u8>().add(self.offset), self.len) }
    }
}

impl ops::DerefMut for AlignedSlice {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            slice::from_raw_parts_mut(
                self.buf.as_mut_ptr().cast::<u8>().add(self.offset),
                self.len,
            )
        }
    }
}

#[test]
fn test_alignment() {
    let v = AlignedSlice::new_zeroed(1, 0);
    assert_eq!(v.len(), 1);
    assert_eq!(v.as_ptr().addr() % PAGE_SIZE, 0);

    let v = AlignedSlice::new_zeroed(PAGE_SIZE + 1, 0);
    assert_eq!(v.len(), PAGE_SIZE + 1);
    assert_eq!(v.as_ptr().addr() % PAGE_SIZE, 0);

    let v = AlignedSlice::new_zeroed(1, 1);
    assert_eq!(v.len(), 1);
    assert_eq!(v.as_ptr().addr() % 2, 1);

    let v = AlignedSlice::new_zeroed(1, 64);
    assert_eq!(v.len(), 1);
    assert_eq!(v.as_ptr().addr() % 64, 0);
    assert_eq!(v.as_ptr().addr() % 128, 64);
}
