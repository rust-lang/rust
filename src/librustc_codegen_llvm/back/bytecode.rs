//! Management of the encoding of LLVM bytecode into rlibs
//!
//! This module contains the management of encoding LLVM bytecode into rlibs,
//! primarily for the usage in LTO situations. Currently the compiler will
//! unconditionally encode LLVM-IR into rlibs regardless of what's happening
//! elsewhere, so we currently compress the bytecode via deflate to avoid taking
//! up too much space on disk.
//!
//! After compressing the bytecode we then have the rest of the format to
//! basically deal with various bugs in various archive implementations. The
//! format currently is:
//!
//!     RLIB LLVM-BYTECODE OBJECT LAYOUT
//!     Version 2
//!     Bytes    Data
//!     0..10    "RUST_OBJECT" encoded in ASCII
//!     11..14   format version as little-endian u32
//!     15..19   the length of the module identifier string
//!     20..n    the module identifier string
//!     n..n+8   size in bytes of deflate compressed LLVM bitcode as
//!              little-endian u64
//!     n+9..    compressed LLVM bitcode
//!     ?        maybe a byte to make this whole thing even length

use std::io::{Read, Write};
use std::ptr;
use std::str;

use flate2::Compression;
use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;

// This is the "magic number" expected at the beginning of a LLVM bytecode
// object in an rlib.
pub const RLIB_BYTECODE_OBJECT_MAGIC: &[u8] = b"RUST_OBJECT";

// The version number this compiler will write to bytecode objects in rlibs
pub const RLIB_BYTECODE_OBJECT_VERSION: u8 = 2;

pub fn encode(identifier: &str, bytecode: &[u8]) -> Vec<u8> {
    let mut encoded = Vec::new();

    // Start off with the magic string
    encoded.extend_from_slice(RLIB_BYTECODE_OBJECT_MAGIC);

    // Next up is the version
    encoded.extend_from_slice(&[RLIB_BYTECODE_OBJECT_VERSION, 0, 0, 0]);

    // Next is the LLVM module identifier length + contents
    let identifier_len = identifier.len();
    encoded.extend_from_slice(&[
        (identifier_len >>  0) as u8,
        (identifier_len >>  8) as u8,
        (identifier_len >> 16) as u8,
        (identifier_len >> 24) as u8,
    ]);
    encoded.extend_from_slice(identifier.as_bytes());

    // Next is the LLVM module deflate compressed, prefixed with its length. We
    // don't know its length yet, so fill in 0s
    let deflated_size_pos = encoded.len();
    encoded.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);

    let before = encoded.len();
    DeflateEncoder::new(&mut encoded, Compression::fast())
        .write_all(bytecode)
        .unwrap();
    let after = encoded.len();

    // Fill in the length we reserved space for before
    let bytecode_len = (after - before) as u64;
    encoded[deflated_size_pos + 0] = (bytecode_len >>  0) as u8;
    encoded[deflated_size_pos + 1] = (bytecode_len >>  8) as u8;
    encoded[deflated_size_pos + 2] = (bytecode_len >> 16) as u8;
    encoded[deflated_size_pos + 3] = (bytecode_len >> 24) as u8;
    encoded[deflated_size_pos + 4] = (bytecode_len >> 32) as u8;
    encoded[deflated_size_pos + 5] = (bytecode_len >> 40) as u8;
    encoded[deflated_size_pos + 6] = (bytecode_len >> 48) as u8;
    encoded[deflated_size_pos + 7] = (bytecode_len >> 56) as u8;

    // If the number of bytes written to the object so far is odd, add a
    // padding byte to make it even. This works around a crash bug in LLDB
    // (see issue #15950)
    if encoded.len() % 2 == 1 {
        encoded.push(0);
    }

    return encoded
}

pub struct DecodedBytecode<'a> {
    identifier: &'a str,
    encoded_bytecode: &'a [u8],
}

impl<'a> DecodedBytecode<'a> {
    pub fn new(data: &'a [u8]) -> Result<DecodedBytecode<'a>, &'static str> {
        if !data.starts_with(RLIB_BYTECODE_OBJECT_MAGIC) {
            return Err("magic bytecode prefix not found")
        }
        let data = &data[RLIB_BYTECODE_OBJECT_MAGIC.len()..];
        if !data.starts_with(&[RLIB_BYTECODE_OBJECT_VERSION, 0, 0, 0]) {
            return Err("wrong version prefix found in bytecode")
        }
        let data = &data[4..];
        if data.len() < 4 {
            return Err("bytecode corrupted")
        }
        let identifier_len = unsafe {
            u32::from_le(ptr::read_unaligned(data.as_ptr() as *const u32)) as usize
        };
        let data = &data[4..];
        if data.len() < identifier_len {
            return Err("bytecode corrupted")
        }
        let identifier = match str::from_utf8(&data[..identifier_len]) {
            Ok(s) => s,
            Err(_) => return Err("bytecode corrupted")
        };
        let data = &data[identifier_len..];
        if data.len() < 8 {
            return Err("bytecode corrupted")
        }
        let bytecode_len = unsafe {
            u64::from_le(ptr::read_unaligned(data.as_ptr() as *const u64)) as usize
        };
        let data = &data[8..];
        if data.len() < bytecode_len {
            return Err("bytecode corrupted")
        }
        let encoded_bytecode = &data[..bytecode_len];

        Ok(DecodedBytecode {
            identifier,
            encoded_bytecode,
        })
    }

    pub fn bytecode(&self) -> Vec<u8> {
        let mut data = Vec::new();
        DeflateDecoder::new(self.encoded_bytecode).read_to_end(&mut data).unwrap();
        return data
    }

    pub fn identifier(&self) -> &'a str {
        self.identifier
    }
}
