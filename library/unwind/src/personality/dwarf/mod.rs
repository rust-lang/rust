//! Utilities for parsing DWARF-encoded data streams.
//! See <http://www.dwarfstd.org>,
//! DWARF-4 standard, Section 7 - "Data Representation"

// This module is used only by x86_64-pc-windows-gnu for now, but we
// are compiling it everywhere to avoid regressions.
#![allow(unused)]
#![forbid(unsafe_op_in_unsafe_fn)]

#[cfg(test)]
mod tests;

pub mod eh;

pub struct DwarfReader {
    pub ptr: *const u8,
}

impl DwarfReader {
    pub fn new(ptr: *const u8) -> DwarfReader {
        DwarfReader { ptr }
    }

    /// Read a type T and then bump the pointer by that amount.
    ///
    /// DWARF streams are "packed", so all types must be read at align 1.
    pub unsafe fn read<T: Copy>(&mut self) -> T {
        unsafe {
            let result = self.ptr.cast::<T>().read_unaligned();
            self.ptr = self.ptr.byte_add(size_of::<T>());
            result
        }
    }

    /// ULEB128 and SLEB128 encodings are defined in Section 7.6 - "Variable Length Data".
    pub unsafe fn read_uleb128(&mut self) -> u64 {
        let mut shift: usize = 0;
        let mut result: u64 = 0;
        let mut byte: u8;
        loop {
            byte = unsafe { self.read::<u8>() };
            result |= ((byte & 0x7F) as u64) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                break;
            }
        }
        result
    }

    pub unsafe fn read_sleb128(&mut self) -> i64 {
        let mut shift: u32 = 0;
        let mut result: u64 = 0;
        let mut byte: u8;
        loop {
            byte = unsafe { self.read::<u8>() };
            result |= ((byte & 0x7F) as u64) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                break;
            }
        }
        // sign-extend
        if shift < u64::BITS && (byte & 0x40) != 0 {
            result |= (!0 as u64) << shift;
        }
        result as i64
    }
}
