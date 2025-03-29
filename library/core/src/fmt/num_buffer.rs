use crate::mem::MaybeUninit;
use crate::slice::SliceIndex;

/// 40 is chosen as the buffer length, as it is equal
/// to that required to accommodate i128::MIN, which has the largest
/// decimal string representation
/// (39 decimal digits + 1 for negative sign).
const BUF_SIZE: usize = 40;

/// A minimal buffer implementation containing elements of type
/// `MaybeUninit<u8>`.
#[unstable(feature = "int_format_into", issue = "138215")]
#[derive(Debug)]
pub struct NumBuffer {
    /// An array of elements of type `MaybeUninit<u8>`.
    contents: [MaybeUninit<u8>; BUF_SIZE],
}

#[unstable(feature = "int_format_into", issue = "138215")]
impl NumBuffer {
    /// Initializes `contents` as an uninitialized array of `MaybeUninit<u8>`.
    #[unstable(feature = "int_format_into", issue = "138215")]
    pub fn new() -> Self {
        NumBuffer { contents: [MaybeUninit::<u8>::uninit(); BUF_SIZE] }
    }

    /// Returns the length of the buffer.
    #[unstable(feature = "int_format_into", issue = "138215")]
    pub fn len(&self) -> usize {
        BUF_SIZE
    }

    /// Extracts a slice of the contents of the buffer.
    /// This function is unsafe, since it does not itself
    /// bounds-check `index`.
    ///
    /// SAFETY: `index` is bounds-checked by the caller.
    #[unstable(feature = "int_format_into", issue = "138215")]
    pub(crate) unsafe fn extract<I>(&self, index: I) -> &I::Output
    where
        I: SliceIndex<[MaybeUninit<u8>]>,
    {
        // SAFETY: `index` is bound-checked by the caller.
        unsafe { self.contents.get_unchecked(index) }
    }

    /// Returns a mutable pointer pointing to the start of the buffer.
    #[unstable(feature = "int_format_into", issue = "138215")]
    #[cfg(feature = "optimize_for_size")]
    pub(crate) fn extract_start_mut_ptr(buf: &mut Self) -> *mut u8 {
        MaybeUninit::slice_as_mut_ptr(&mut buf.contents)
    }

    /// Writes data at index `offset` of the buffer.
    /// This function is unsafe, since it does not itself perform
    /// the safety checks below.
    ///
    /// SAFETY: The caller ensures the following:
    /// 1. `offset` is bounds-checked.
    /// 2. `data` is a valid ASCII character.
    #[unstable(feature = "int_format_into", issue = "138215")]
    pub(crate) unsafe fn write(&mut self, offset: usize, data: u8) {
        self.contents[offset].write(data);
    }
}
