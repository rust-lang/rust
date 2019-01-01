//! Utilities for parsing DWARF-encoded data streams.
//! See <http://www.dwarfstd.org>,
//! DWARF-4 standard, Section 7 - "Data Representation"

// This module is used only by x86_64-pc-windows-gnu for now, but we
// are compiling it everywhere to avoid regressions.
#![allow(unused)]

pub mod eh;

use core::mem;

pub struct DwarfReader {
    pub ptr: *const u8,
}

#[repr(C,packed)]
struct Unaligned<T>(T);

impl DwarfReader {
    pub fn new(ptr: *const u8) -> DwarfReader {
        DwarfReader { ptr }
    }

    // DWARF streams are packed, so e.g., a u32 would not necessarily be aligned
    // on a 4-byte boundary. This may cause problems on platforms with strict
    // alignment requirements. By wrapping data in a "packed" struct, we are
    // telling the backend to generate "misalignment-safe" code.
    pub unsafe fn read<T: Copy>(&mut self) -> T {
        let Unaligned(result) = *(self.ptr as *const Unaligned<T>);
        self.ptr = self.ptr.add(mem::size_of::<T>());
        result
    }

    // ULEB128 and SLEB128 encodings are defined in Section 7.6 - "Variable
    // Length Data".
    pub unsafe fn read_uleb128(&mut self) -> u64 {
        let mut shift: usize = 0;
        let mut result: u64 = 0;
        let mut byte: u8;
        loop {
            byte = self.read::<u8>();
            result |= ((byte & 0x7F) as u64) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                break;
            }
        }
        result
    }

    pub unsafe fn read_sleb128(&mut self) -> i64 {
        let mut shift: usize = 0;
        let mut result: u64 = 0;
        let mut byte: u8;
        loop {
            byte = self.read::<u8>();
            result |= ((byte & 0x7F) as u64) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                break;
            }
        }
        // sign-extend
        if shift < 8 * mem::size_of::<u64>() && (byte & 0x40) != 0 {
            result |= (!0 as u64) << shift;
        }
        result as i64
    }
}

#[test]
fn dwarf_reader() {
    let encoded: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 0xE5, 0x8E, 0x26, 0x9B, 0xF1, 0x59, 0xFF, 0xFF];

    let mut reader = DwarfReader::new(encoded.as_ptr());

    unsafe {
        assert!(reader.read::<u8>() == u8::to_be(1u8));
        assert!(reader.read::<u16>() == u16::to_be(0x0203));
        assert!(reader.read::<u32>() == u32::to_be(0x04050607));

        assert!(reader.read_uleb128() == 624485);
        assert!(reader.read_sleb128() == -624485);

        assert!(reader.read::<i8>() == i8::to_be(-1));
    }
}
