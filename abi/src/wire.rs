//! Wire-safe primitives and IDs for the Graphable contract.
//!
//! This module defines the fundamental fixed-size IDs used in the Thing-OS
//! data graph and the `WireSafe` trait used to enforce pointer-free,
//! packed layouts for payload structs.

use core::marker::PhantomData;
use core::sync::atomic::AtomicU64;
#[cfg(feature = "kernel-id-gen")]
use core::sync::atomic::Ordering;

/// 128-bit unique identifier for a Thing in the graph.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ThingId(pub [u8; 16]);

/// 128-bit content-addressable identifier for a Blob (large binary data).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct BlobId(pub [u8; 16]);

/// 128-bit identifier for an interned Symbol (string).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SymbolId(pub [u8; 16]);

/// 128-bit identifier for a Schema Kind, derived from the stable hash of the Schema.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct KindId(pub [u8; 16]);

/// 128-bit identifier for a Predicate (edge type).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct PredicateId(pub [u8; 16]);

#[cfg(feature = "kernel-id-gen")]
static NEXT_THING_ID: AtomicU64 = AtomicU64::new(0);

impl ThingId {
    #[cfg(feature = "kernel-id-gen")]
    pub fn new_debug_nonce() -> Self {
        let seq = NEXT_THING_ID.fetch_add(1, Ordering::Relaxed);
        // Simple nonce to distinguish runs (if ASLR/pointers vary)
        let bootish = (core::ptr::addr_of!(NEXT_THING_ID) as u64) ^ 0x5EED_C0DE_CAFE_BABE;

        let mut bytes = [0u8; 16];
        bytes[0..8].copy_from_slice(&bootish.to_le_bytes());
        bytes[8..16].copy_from_slice(&seq.to_le_bytes());
        let id = Self(bytes);

        #[cfg(debug_assertions)]
        {
            // Simple tripwire to catch collisions in debug builds.
            // Uses a small ring buffer and a spinlock.
            static mut DEBUG_HISTORY: [ThingId; 1024] = [ThingId([0; 16]); 1024];
            static mut DEBUG_CURSOR: usize = 0;
            static DEBUG_LOCK: core::sync::atomic::AtomicBool =
                core::sync::atomic::AtomicBool::new(false);

            while DEBUG_LOCK
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                core::hint::spin_loop();
            }

            // SAFETY: We hold the spinlock, so we have exclusive access to the static muts.
            unsafe {
                for i in 0..1024 {
                    // Ignore zero-initialized slots if our ID is non-zero (which it should be).
                    // If we somehow generated a zero ID, we'd panic on the first run, which is also good (bug).
                    if DEBUG_HISTORY[i] == id {
                        panic!("ThingId collision detected: {:?}", id);
                    }
                }
                DEBUG_HISTORY[DEBUG_CURSOR] = id;
                DEBUG_CURSOR = (DEBUG_CURSOR + 1) % 1024;
            }

            DEBUG_LOCK.store(false, Ordering::Release);
        }

        id
    }
}

impl PredicateId {
    /// Lossy conversion of a 128-bit PredicateId to a u32 (little-endian).
    pub fn to_u32_lossy(self) -> u32 {
        u32::from_le_bytes(self.0[0..4].try_into().unwrap())
    }
}

impl SymbolId {
    /// Convert a BlobId to a SymbolId.
    ///
    /// In this implementation, we simply hash the BlobId to get a SymbolId,
    /// or treating it as distinct type-safe handle.
    /// For now, since they are both 16 bytes, we can map 1:1 if we want,
    /// but let's do a quick mix to ensure they are distinct spaces if needed.
    /// Actually, the prompt suggests "stable hash of the BlobId bytes" or simple mapping.
    /// Let's use blake3 hashing of the BlobId bytes to be consistent with "derived" IDs.
    pub fn from_blob(blob: BlobId) -> SymbolId {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"SymbolId");
        hasher.update(&blob.0);
        let hash = hasher.finalize();
        let mut out = [0u8; 16];
        out.copy_from_slice(&hash.as_bytes()[0..16]);
        SymbolId(out)
    }
}

/// Marker trait for types that are safe to transmit over the wire (pointer-free, packed, fixed-size).
///
/// # Safety
///
/// Implementing this trait asserts that:
/// 1. The type is `Copy` and `'static`.
/// 2. The type contains NO pointers, references, `Box`, `Vec`, `String`, etc.
/// 3. The type has a stable, platform-independent memory layout (when used with `encode_packed_le`).
///    Ideally `#[repr(C)]` or `#[repr(C, packed)]`.
pub unsafe trait WireSafe: Copy + 'static {}

// Null implementation for PhantomData generic over WireSafe types
unsafe impl<T: WireSafe> WireSafe for PhantomData<T> {}

// Primitive implementations
unsafe impl WireSafe for u8 {}
unsafe impl WireSafe for u16 {}
unsafe impl WireSafe for u32 {}
unsafe impl WireSafe for u64 {}
unsafe impl WireSafe for u128 {}
unsafe impl WireSafe for i8 {}
unsafe impl WireSafe for i16 {}
unsafe impl WireSafe for i32 {}
unsafe impl WireSafe for i64 {}
unsafe impl WireSafe for i128 {}
unsafe impl WireSafe for f32 {}
unsafe impl WireSafe for f64 {}

// IDs
unsafe impl WireSafe for ThingId {}
unsafe impl WireSafe for BlobId {}
unsafe impl WireSafe for SymbolId {}
unsafe impl WireSafe for KindId {}
unsafe impl WireSafe for PredicateId {}

// Arrays
unsafe impl<T: WireSafe, const N: usize> WireSafe for [T; N] {}

/// Assert that a type is WireSafe at compile time.
///
/// Used in impl blocks to enforce constraints.
pub const fn assert_wire_safe<T: WireSafe>() {}
