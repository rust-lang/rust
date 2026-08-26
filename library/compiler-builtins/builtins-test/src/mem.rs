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

pub mod mcpy {
    use super::*;

    pub struct Cfg {
        pub len: usize,
        pub s_off: usize,
        pub d_off: usize,
    }

    /// Return `(len, src, dst)` for a cfg.
    pub fn setup(cfg: Cfg) -> (usize, AlignedSlice, AlignedSlice) {
        let Cfg { len, s_off, d_off } = cfg;
        let mut src = AlignedSlice::new_zeroed(len, s_off);
        let dst = AlignedSlice::new_zeroed(len, d_off);
        src.fill(1);
        (len, src, dst)
    }
}

pub mod mset {
    use super::*;

    pub struct Cfg {
        pub len: usize,
        pub offset: usize,
    }

    pub fn setup(Cfg { len, offset }: Cfg) -> (usize, AlignedSlice) {
        (len, AlignedSlice::new_zeroed(len, offset))
    }
}

pub mod mcmp {
    use super::*;

    pub struct Cfg {
        pub len: usize,
        pub s_off: usize,
        pub d_off: usize,
    }

    pub fn setup(cfg: Cfg) -> (usize, AlignedSlice, AlignedSlice) {
        let Cfg { len, s_off, d_off } = cfg;
        let b1 = AlignedSlice::new_zeroed(len, s_off);
        let mut b2 = AlignedSlice::new_zeroed(len, d_off);
        b2[len - 1] = 1;
        (len, b1, b2)
    }
}

pub mod mmove {
    use Spread::{Aligned, Large, Medium, Small};

    use super::*;

    pub struct Cfg {
        pub len: usize,
        pub spread: Spread,
        pub off: usize,
    }

    pub enum Spread {
        /// `src` and `dst` are close and have the same alignment (or offset).
        Aligned,
        /// `src` and `dst` are close.
        Small,
        /// `src` and `dst` are halfway offset in the buffer.
        Medium,
        /// `src` and `dst` only overlap by a single byte.
        Large,
    }

    // Note that small and large are
    pub fn calculate_spread(len: usize, spread: Spread) -> usize {
        match spread {
            // Note that this test doesn't make sense for lengths less than len=128
            Aligned => {
                assert!(
                    len > MAX_TESTED_ALIGN,
                    "aligned memset would have no overlap"
                );
                MAX_TESTED_ALIGN
            }
            Small => 1,
            Medium => (len / 2) + 1, // add 1 so all are misaligned
            Large => len - 1,
        }
    }

    pub fn setup_forward(cfg: Cfg) -> (usize, usize, AlignedSlice) {
        let Cfg { len, spread, off } = cfg;
        let spread = calculate_spread(len, spread);
        assert!(spread < len, "memmove tests should have some overlap");
        let mut buf = AlignedSlice::new_zeroed(len + spread, off);
        let mut fill: usize = 0;
        buf[..len].fill_with(|| {
            fill += 1;
            fill as u8
        });
        (len, spread, buf)
    }

    pub fn setup_backward(cfg: Cfg) -> (usize, usize, AlignedSlice) {
        let Cfg { len, spread, off } = cfg;
        let spread = calculate_spread(len, spread);
        assert!(spread < len, "memmove tests should have some overlap");
        let mut buf = AlignedSlice::new_zeroed(len + spread, off);
        let mut fill: usize = 0;
        buf[spread..].fill_with(|| {
            fill += 1;
            fill as u8
        });
        (len, spread, buf)
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
