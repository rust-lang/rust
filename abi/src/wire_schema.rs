//! Language-neutral schema description for Thing-OS Graphable types.
//!
//! This module defines the `Schema` struct and `WireType` enum used to describe
//! the layout of packed data structures. It provides the `schema_hash` function
//! (using blake3) to generate a stable `KindId` from a schema definition.

use crate::wire::KindId;

/// Canonical schema description for a Graphable type.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Schema {
    pub name: &'static str,
    pub fields: &'static [Field],
}

/// A field in a schema.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Field {
    pub name: &'static str,
    pub ty: WireType,
}

/// The wire type of a field.
///
/// This enum is designed to be `Copy` and `const`-constructible (no Boxes).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WireType {
    U8,
    U16,
    U32,
    U64,
    U128,
    I8,
    I16,
    I32,
    I64,
    I128,
    F32,
    F64,
    Bool, // if we decide to use it, treated as u8
    ThingId,
    BlobId,
    SymbolId,
    KindId,
    /// Fixed-size array, e.g. [T; N]
    Array(&'static WireType, usize),
    /// Nested struct (flat, packed)
    Struct(&'static Schema),
    /// Helper for "Bytes" which is Array(U8, N)
    Bytes(usize),
}

impl WireType {
    /// Return the size in bytes of this type on the wire (packed).
    pub fn size(&self) -> usize {
        match self {
            WireType::U8 | WireType::I8 | WireType::Bool => 1,
            WireType::U16 | WireType::I16 => 2,
            WireType::U32 | WireType::I32 | WireType::F32 => 4,
            WireType::U64 | WireType::I64 | WireType::F64 => 8,
            WireType::U128 | WireType::I128 => 16,
            WireType::ThingId | WireType::BlobId | WireType::SymbolId | WireType::KindId => 16,
            WireType::Array(t, n) => t.size() * n,
            WireType::Struct(s) => s.size(),
            WireType::Bytes(n) => *n,
        }
    }
}

impl Schema {
    pub fn size(&self) -> usize {
        let mut sum = 0;
        let mut i = 0;
        while i < self.fields.len() {
            sum += self.fields[i].ty.size();
            i += 1;
        }
        sum
    }
}

/// Compute the stable hash of a Schema to generate a KindId.
///
/// Rules:
/// - Canonical encoding of fields in order.
/// - Names utf-8 bytes.
/// - Types with fixed tags.
pub fn schema_hash(schema: &Schema) -> KindId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ThingOS-Schema-v1");
    // Mix name
    hasher.update(&(schema.name.len() as u64).to_le_bytes());
    hasher.update(schema.name.as_bytes());

    // Mix fields
    hasher.update(&(schema.fields.len() as u64).to_le_bytes());
    for field in schema.fields {
        // Name
        hasher.update(&(field.name.len() as u64).to_le_bytes());
        hasher.update(field.name.as_bytes());
        // Type
        hash_wire_type(&mut hasher, &field.ty);
    }

    let hash = hasher.finalize();
    let mut out = [0u8; 16];
    out.copy_from_slice(&hash.as_bytes()[0..16]);
    KindId(out)
}

fn hash_wire_type(hasher: &mut blake3::Hasher, ty: &WireType) {
    match ty {
        WireType::U8 => {
            hasher.update(&[1]);
        }
        WireType::U16 => {
            hasher.update(&[2]);
        }
        WireType::U32 => {
            hasher.update(&[3]);
        }
        WireType::U64 => {
            hasher.update(&[4]);
        }
        WireType::U128 => {
            hasher.update(&[5]);
        }
        WireType::I8 => {
            hasher.update(&[6]);
        }
        WireType::I16 => {
            hasher.update(&[7]);
        }
        WireType::I32 => {
            hasher.update(&[8]);
        }
        WireType::I64 => {
            hasher.update(&[9]);
        }
        WireType::I128 => {
            hasher.update(&[10]);
        }
        WireType::F32 => {
            hasher.update(&[11]);
        }
        WireType::F64 => {
            hasher.update(&[12]);
        }
        WireType::Bool => {
            hasher.update(&[13]);
        }
        WireType::ThingId => {
            hasher.update(&[14]);
        }
        WireType::BlobId => {
            hasher.update(&[15]);
        }
        WireType::SymbolId => {
            hasher.update(&[16]);
        }
        WireType::KindId => {
            hasher.update(&[17]);
        }
        WireType::Array(inner, len) => {
            hasher.update(&[18]);
            hasher.update(&(*len as u64).to_le_bytes());
            hash_wire_type(hasher, inner);
        }
        WireType::Struct(s) => {
            hasher.update(&[19]);
            // Recurse? If we include structs, we should hash their name/fields too.
            // But to avoid infinite loops if recursive (though static refs prevent runtime cycles, logic cycles possible?),
            // let's just hash the definition.
            // Wait, infinite recursion in types isn't possible with Sized structs in Rust unless generic/pointer (which we banned).
            // So recursion is fine.
            hasher.update(&(s.name.len() as u64).to_le_bytes());
            hasher.update(s.name.as_bytes());
            hasher.update(&(s.fields.len() as u64).to_le_bytes());
            for f in s.fields {
                hasher.update(&(f.name.len() as u64).to_le_bytes());
                hasher.update(f.name.as_bytes());
                hash_wire_type(hasher, &f.ty);
            }
        }
        WireType::Bytes(len) => {
            hasher.update(&[20]);
            hasher.update(&(*len as u64).to_le_bytes());
        }
    }
}
