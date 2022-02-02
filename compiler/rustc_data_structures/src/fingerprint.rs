use crate::stable_hasher;
use rustc_serialize::{Decodable, Encodable};
use std::convert::TryInto;
use std::hash::{Hash, Hasher};

#[cfg(test)]
mod tests;

#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Clone, Copy)]
#[repr(C)]
pub struct Fingerprint(u64, u64);

impl Fingerprint {
    pub const ZERO: Fingerprint = Fingerprint(0, 0);

    #[inline]
    pub fn new(_0: u64, _1: u64) -> Fingerprint {
        Fingerprint(_0, _1)
    }

    #[inline]
    pub fn from_smaller_hash(hash: u64) -> Fingerprint {
        Fingerprint(hash, hash)
    }

    #[inline]
    pub fn to_smaller_hash(&self) -> u64 {
        // Even though both halves of the fingerprint are expected to be good
        // quality hash values, let's still combine the two values because the
        // Fingerprints in DefPathHash have the StableCrateId portion which is
        // the same for all DefPathHashes from the same crate. Combining the
        // two halfs makes sure we get a good quality hash in such cases too.
        self.0.wrapping_mul(3).wrapping_add(self.1)
    }

    #[inline]
    pub fn as_value(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    #[inline]
    pub fn combine(self, other: Fingerprint) -> Fingerprint {
        // See https://stackoverflow.com/a/27952689 on why this function is
        // implemented this way.
        Fingerprint(
            self.0.wrapping_mul(3).wrapping_add(other.0),
            self.1.wrapping_mul(3).wrapping_add(other.1),
        )
    }

    // Combines two hashes in an order independent way. Make sure this is what
    // you want.
    #[inline]
    pub fn combine_commutative(self, other: Fingerprint) -> Fingerprint {
        let a = u128::from(self.1) << 64 | u128::from(self.0);
        let b = u128::from(other.1) << 64 | u128::from(other.0);

        let c = a.wrapping_add(b);

        Fingerprint(c as u64, (c >> 64) as u64)
    }

    pub fn to_hex(&self) -> String {
        format!("{:x}{:x}", self.0, self.1)
    }

    #[inline]
    pub fn to_le_bytes(&self) -> [u8; 16] {
        // This seems to optimize to the same machine code as
        // `unsafe { mem::transmute(*k) }`. Well done, LLVM! :)
        let mut result = [0u8; 16];

        let first_half: &mut [u8; 8] = (&mut result[0..8]).try_into().unwrap();
        *first_half = self.0.to_le_bytes();

        let second_half: &mut [u8; 8] = (&mut result[8..16]).try_into().unwrap();
        *second_half = self.1.to_le_bytes();

        result
    }

    #[inline]
    pub fn from_le_bytes(bytes: [u8; 16]) -> Fingerprint {
        Fingerprint(
            u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
        )
    }
}

impl std::fmt::Display for Fingerprint {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:x}-{:x}", self.0, self.1)
    }
}

impl Hash for Fingerprint {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_fingerprint(self);
    }
}

trait FingerprintHasher {
    fn write_fingerprint(&mut self, fingerprint: &Fingerprint);
}

impl<H: Hasher> FingerprintHasher for H {
    #[inline]
    default fn write_fingerprint(&mut self, fingerprint: &Fingerprint) {
        self.write_u64(fingerprint.0);
        self.write_u64(fingerprint.1);
    }
}

impl FingerprintHasher for crate::unhash::Unhasher {
    #[inline]
    fn write_fingerprint(&mut self, fingerprint: &Fingerprint) {
        // Even though both halves of the fingerprint are expected to be good
        // quality hash values, let's still combine the two values because the
        // Fingerprints in DefPathHash have the StableCrateId portion which is
        // the same for all DefPathHashes from the same crate. Combining the
        // two halfs makes sure we get a good quality hash in such cases too.
        //
        // Since `Unhasher` is used only in the context of HashMaps, it is OK
        // to combine the two components in an order-independent way (which is
        // cheaper than the more robust Fingerprint::to_smaller_hash()). For
        // HashMaps we don't really care if Fingerprint(x,y) and
        // Fingerprint(y, x) result in the same hash value. Collision
        // probability will still be much better than with FxHash.
        self.write_u64(fingerprint.0.wrapping_add(fingerprint.1));
    }
}

impl stable_hasher::StableHasherResult for Fingerprint {
    #[inline]
    fn finish(hasher: stable_hasher::StableHasher) -> Self {
        let (_0, _1) = hasher.finalize();
        Fingerprint(_0, _1)
    }
}

impl_stable_hash_via_hash!(Fingerprint);

impl<E: rustc_serialize::Encoder> Encodable<E> for Fingerprint {
    #[inline]
    fn encode(&self, s: &mut E) -> Result<(), E::Error> {
        s.emit_raw_bytes(&self.to_le_bytes())?;
        Ok(())
    }
}

impl<D: rustc_serialize::Decoder> Decodable<D> for Fingerprint {
    #[inline]
    fn decode(d: &mut D) -> Self {
        let mut bytes = [0u8; 16];
        d.read_raw_bytes_into(&mut bytes);
        Fingerprint::from_le_bytes(bytes)
    }
}

// `PackedFingerprint` wraps a `Fingerprint`. Its purpose is to, on certain
// architectures, behave like a `Fingerprint` without alignment requirements.
// This behavior is only enabled on x86 and x86_64, where the impact of
// unaligned accesses is tolerable in small doses.
//
// This may be preferable to use in large collections of structs containing
// fingerprints, as it can reduce memory consumption by preventing the padding
// that the more strictly-aligned `Fingerprint` can introduce. An application of
// this is in the query dependency graph, which contains a large collection of
// `DepNode`s. As of this writing, the size of a `DepNode` decreases by ~30%
// (from 24 bytes to 17) by using the packed representation here, which
// noticeably decreases total memory usage when compiling large crates.
//
// The wrapped `Fingerprint` is private to reduce the chance of a client
// invoking undefined behavior by taking a reference to the packed field.
#[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), repr(packed))]
#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Clone, Copy, Hash)]
pub struct PackedFingerprint(Fingerprint);

impl std::fmt::Display for PackedFingerprint {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Copy to avoid taking reference to packed field.
        let copy = self.0;
        copy.fmt(formatter)
    }
}

impl<E: rustc_serialize::Encoder> Encodable<E> for PackedFingerprint {
    #[inline]
    fn encode(&self, s: &mut E) -> Result<(), E::Error> {
        // Copy to avoid taking reference to packed field.
        let copy = self.0;
        copy.encode(s)
    }
}

impl<D: rustc_serialize::Decoder> Decodable<D> for PackedFingerprint {
    #[inline]
    fn decode(d: &mut D) -> Self {
        Self(Fingerprint::decode(d))
    }
}

impl From<Fingerprint> for PackedFingerprint {
    #[inline]
    fn from(f: Fingerprint) -> PackedFingerprint {
        PackedFingerprint(f)
    }
}

impl From<PackedFingerprint> for Fingerprint {
    #[inline]
    fn from(f: PackedFingerprint) -> Fingerprint {
        f.0
    }
}
