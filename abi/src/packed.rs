//! Packed Little-Endian Codec for WireSafe types using Schema.
//!
//! This module provides the `encode_schema` and `decode_schema` functions which
//! walk a `Schema` to serializing/deserializing data, ensuring correct endianness (Little Endian)
//! on the wire regardless of host architecture.

use crate::errors::{Error, Result};
use crate::wire::WireSafe;
use crate::wire_schema::{Schema, WireType};
use core::ptr;

/// Encode a value described by `schema` from `src` to `out`, enforcing Little Endian wire format.
///
/// `src` must point to the start of the Rust struct.
/// `out` is the destination buffer.
/// Returns number of bytes written.
///
/// # Safety
/// Caller must ensure `src` points to a valid instance of the type described by `schema`.
pub unsafe fn encode_schema(src: *const u8, schema: &Schema, out: &mut [u8]) -> Result<usize> {
    if out.len() < schema.size() {
        return Err(Error::BufferTooSmall {
            required: schema.size(),
            available: out.len(),
        });
    }

    let mut cursor = 0;
    // We assume the input struct is packed (#[repr(C, packed)]), so fields are contiguous.
    let mut src_offset = 0;

    for field in schema.fields {
        let written = encode_wire_type(src.add(src_offset), &field.ty, &mut out[cursor..])?;
        cursor += written;
        src_offset += written; // Packed struct assumption
    }

    Ok(cursor)
}

unsafe fn encode_wire_type(src: *const u8, ty: &WireType, out: &mut [u8]) -> Result<usize> {
    match ty {
        WireType::U8 | WireType::I8 | WireType::Bool => {
            out[0] = *src;
            Ok(1)
        }
        WireType::U16 => {
            let val = ptr::read_unaligned(src as *const u16);
            out[0..2].copy_from_slice(&val.to_le_bytes());
            Ok(2)
        }
        WireType::I16 => {
            let val = ptr::read_unaligned(src as *const i16);
            out[0..2].copy_from_slice(&val.to_le_bytes());
            Ok(2)
        }
        WireType::U32 => {
            let val = ptr::read_unaligned(src as *const u32);
            out[0..4].copy_from_slice(&val.to_le_bytes());
            Ok(4)
        }
        WireType::I32 => {
            let val = ptr::read_unaligned(src as *const i32);
            out[0..4].copy_from_slice(&val.to_le_bytes());
            Ok(4)
        }
        WireType::F32 => {
            // treat as u32 bits
            let val = ptr::read_unaligned(src as *const u32);
            out[0..4].copy_from_slice(&val.to_le_bytes());
            Ok(4)
        }
        WireType::U64 => {
            let val = ptr::read_unaligned(src as *const u64);
            out[0..8].copy_from_slice(&val.to_le_bytes());
            Ok(8)
        }
        WireType::I64 => {
            let val = ptr::read_unaligned(src as *const i64);
            out[0..8].copy_from_slice(&val.to_le_bytes());
            Ok(8)
        }
        WireType::F64 => {
            let val = ptr::read_unaligned(src as *const u64);
            out[0..8].copy_from_slice(&val.to_le_bytes());
            Ok(8)
        }
        WireType::U128 => {
            let val = ptr::read_unaligned(src as *const u128);
            out[0..16].copy_from_slice(&val.to_le_bytes());
            Ok(16)
        }
        WireType::I128 => {
            let val = ptr::read_unaligned(src as *const i128);
            out[0..16].copy_from_slice(&val.to_le_bytes());
            Ok(16)
        }
        // IDs are byte arrays, just copy
        WireType::ThingId | WireType::BlobId | WireType::SymbolId | WireType::KindId => {
            out[0..16].copy_from_slice(core::slice::from_raw_parts(src, 16));
            Ok(16)
        }
        WireType::Bytes(len) => {
            ptr::copy_nonoverlapping(src, out.as_mut_ptr(), *len);
            Ok(*len)
        }
        WireType::Array(inner, count) => {
            let mut cursor = 0;
            let mut src_off = 0;
            for _ in 0..*count {
                let w = encode_wire_type(src.add(src_off), inner, &mut out[cursor..])?;
                cursor += w;
                src_off += w;
            }
            Ok(cursor)
        }
        WireType::Struct(s) => {
            // Recursive call
            // We do NOT use encode_schema because encode_schema takes &Schema, and here we have it
            // but logic is same.
            let mut cursor = 0;
            let mut src_off = 0;
            for f in s.fields {
                let w = encode_wire_type(src.add(src_off), &f.ty, &mut out[cursor..])?;
                cursor += w;
                src_off += w;
            }
            Ok(cursor)
        }
    }
}

/// Decode a value described by `schema` from `bytes` to `dst`, enforcing Little Endian wire format.
///
/// `dst` must point to a valid memory location large enough.
pub unsafe fn decode_schema(bytes: &[u8], schema: &Schema, dst: *mut u8) -> Result<usize> {
    if bytes.len() < schema.size() {
        return Err(Error::BufferTooSmall {
            required: schema.size(),
            available: bytes.len(),
        });
    }

    let mut cursor = 0;
    let mut dst_offset = 0;

    for field in schema.fields {
        let read = decode_wire_type(&bytes[cursor..], &field.ty, dst.add(dst_offset))?;
        cursor += read;
        dst_offset += read;
    }

    Ok(cursor)
}

unsafe fn decode_wire_type(bytes: &[u8], ty: &WireType, dst: *mut u8) -> Result<usize> {
    match ty {
        WireType::U8 | WireType::I8 | WireType::Bool => {
            *dst = bytes[0];
            Ok(1)
        }
        WireType::U16 => {
            let arr: [u8; 2] = bytes[0..2].try_into().unwrap(); // safe, checked len
            let val = u16::from_le_bytes(arr);
            ptr::write_unaligned(dst as *mut u16, val);
            Ok(2)
        }
        WireType::I16 => {
            let arr: [u8; 2] = bytes[0..2].try_into().unwrap();
            let val = i16::from_le_bytes(arr);
            ptr::write_unaligned(dst as *mut i16, val);
            Ok(2)
        }
        WireType::U32 => {
            let arr: [u8; 4] = bytes[0..4].try_into().unwrap();
            let val = u32::from_le_bytes(arr);
            ptr::write_unaligned(dst as *mut u32, val);
            Ok(4)
        }
        WireType::I32 => {
            let arr: [u8; 4] = bytes[0..4].try_into().unwrap();
            let val = i32::from_le_bytes(arr);
            ptr::write_unaligned(dst as *mut i32, val);
            Ok(4)
        }
        WireType::F32 => {
            let arr: [u8; 4] = bytes[0..4].try_into().unwrap();
            let val = u32::from_le_bytes(arr); // read as bits
            ptr::write_unaligned(dst as *mut u32, val);
            Ok(4)
        }
        WireType::U64 => {
            let arr: [u8; 8] = bytes[0..8].try_into().unwrap();
            let val = u64::from_le_bytes(arr);
            ptr::write_unaligned(dst as *mut u64, val);
            Ok(8)
        }
        WireType::I64 => {
            let arr: [u8; 8] = bytes[0..8].try_into().unwrap();
            let val = i64::from_le_bytes(arr);
            ptr::write_unaligned(dst as *mut i64, val);
            Ok(8)
        }
        WireType::F64 => {
            let arr: [u8; 8] = bytes[0..8].try_into().unwrap();
            let val = u64::from_le_bytes(arr);
            ptr::write_unaligned(dst as *mut u64, val);
            Ok(8)
        }
        WireType::U128 => {
            let arr: [u8; 16] = bytes[0..16].try_into().unwrap();
            let val = u128::from_le_bytes(arr);
            ptr::write_unaligned(dst as *mut u128, val);
            Ok(16)
        }
        WireType::I128 => {
            let arr: [u8; 16] = bytes[0..16].try_into().unwrap();
            let val = i128::from_le_bytes(arr);
            ptr::write_unaligned(dst as *mut i128, val);
            Ok(16)
        }
        WireType::ThingId | WireType::BlobId | WireType::SymbolId | WireType::KindId => {
            ptr::copy_nonoverlapping(bytes.as_ptr(), dst, 16);
            Ok(16)
        }
        WireType::Bytes(len) => {
            ptr::copy_nonoverlapping(bytes.as_ptr(), dst, *len);
            Ok(*len)
        }
        WireType::Array(inner, count) => {
            let mut cursor = 0;
            let mut dst_off = 0;
            for _ in 0..*count {
                let r = decode_wire_type(&bytes[cursor..], inner, dst.add(dst_off))?;
                cursor += r;
                dst_off += r;
            }
            Ok(cursor)
        }
        WireType::Struct(s) => {
            let mut cursor = 0;
            let mut dst_off = 0;
            for f in s.fields {
                let r = decode_wire_type(&bytes[cursor..], &f.ty, dst.add(dst_off))?;
                cursor += r;
                dst_off += r;
            }
            Ok(cursor)
        }
    }
}

/// Legacy/Simple packed codec (memcpy).
/// Only safe if host is Little Endian and types are POD.
pub fn encode_packed_le<T: WireSafe>(v: &T, out: &mut [u8]) -> Result<usize> {
    // Just forward to schema codec now?
    // No, T: WireSafe doesn't imply T: Graphable (so we don't have SCHEMA).
    // But verify uses this.
    // Let's keep strict check to be safe or deprecated.
    // The prompt says "Implement a schema-driven packed codec".
    // We can keep this for bare WireSafe wrappers if needed, but Graphable should use Schema.
    // We'll leave this as raw copy since WireSafe generally implies primitive/fixed.

    use core::mem;
    let size = mem::size_of::<T>();
    if out.len() < size {
        return Err(Error::BufferTooSmall {
            required: size,
            available: out.len(),
        });
    }
    unsafe {
        ptr::copy_nonoverlapping(v as *const T as *const u8, out.as_mut_ptr(), size);
    }
    Ok(size)
}

pub fn decode_packed_le<T: WireSafe>(bytes: &[u8]) -> Result<T> {
    use core::mem;
    let size = mem::size_of::<T>();
    if bytes.len() < size {
        return Err(Error::BufferTooSmall {
            required: size,
            available: bytes.len(),
        });
    }
    if bytes.len() != size {
        return Err(Error::InvalidDataLength {
            expected: size,
            actual: bytes.len(),
        });
    }
    let mut val = mem::MaybeUninit::<T>::uninit();
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), val.as_mut_ptr() as *mut u8, size);
        Ok(val.assume_init())
    }
}
