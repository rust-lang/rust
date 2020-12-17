use crate::stable_hasher;
use rustc_serialize::{
    opaque::{self, EncodeResult},
    Decodable, Encodable,
};
use std::hash::{Hash, Hasher};
use std::mem::{self, MaybeUninit};

#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Clone, Copy)]
pub struct Fingerprint(u64, u64);

impl Fingerprint {
    pub const ZERO: Fingerprint = Fingerprint(0, 0);

    #[inline]
    pub fn from_smaller_hash(hash: u64) -> Fingerprint {
        Fingerprint(hash, hash)
    }

    #[inline]
    pub fn to_smaller_hash(&self) -> u64 {
        self.0
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

        Fingerprint((c >> 64) as u64, c as u64)
    }

    pub fn to_hex(&self) -> String {
        format!("{:x}{:x}", self.0, self.1)
    }

    pub fn encode_opaque(&self, encoder: &mut opaque::Encoder) -> EncodeResult {
        let bytes: [u8; 16] = unsafe { mem::transmute([self.0.to_le(), self.1.to_le()]) };

        encoder.emit_raw_bytes(&bytes);
        Ok(())
    }

    pub fn decode_opaque(decoder: &mut opaque::Decoder<'_>) -> Result<Fingerprint, String> {
        let mut bytes: [MaybeUninit<u8>; 16] = MaybeUninit::uninit_array();

        decoder.read_raw_bytes(&mut bytes)?;

        let [l, r]: [u64; 2] = unsafe { mem::transmute(bytes) };

        Ok(Fingerprint(u64::from_le(l), u64::from_le(r)))
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
        // `Unhasher` only wants a single `u64`
        self.write_u64(fingerprint.0);
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
    fn encode(&self, s: &mut E) -> Result<(), E::Error> {
        s.encode_fingerprint(self)
    }
}

impl<D: rustc_serialize::Decoder> Decodable<D> for Fingerprint {
    fn decode(d: &mut D) -> Result<Self, D::Error> {
        d.decode_fingerprint()
    }
}

pub trait FingerprintEncoder: rustc_serialize::Encoder {
    fn encode_fingerprint(&mut self, f: &Fingerprint) -> Result<(), Self::Error>;
}

pub trait FingerprintDecoder: rustc_serialize::Decoder {
    fn decode_fingerprint(&mut self) -> Result<Fingerprint, Self::Error>;
}

impl<E: rustc_serialize::Encoder> FingerprintEncoder for E {
    default fn encode_fingerprint(&mut self, _: &Fingerprint) -> Result<(), E::Error> {
        panic!("Cannot encode `Fingerprint` with `{}`", std::any::type_name::<E>());
    }
}

impl FingerprintEncoder for opaque::Encoder {
    fn encode_fingerprint(&mut self, f: &Fingerprint) -> EncodeResult {
        f.encode_opaque(self)
    }
}

impl<D: rustc_serialize::Decoder> FingerprintDecoder for D {
    default fn decode_fingerprint(&mut self) -> Result<Fingerprint, D::Error> {
        panic!("Cannot decode `Fingerprint` with `{}`", std::any::type_name::<D>());
    }
}

impl FingerprintDecoder for opaque::Decoder<'_> {
    fn decode_fingerprint(&mut self) -> Result<Fingerprint, String> {
        Fingerprint::decode_opaque(self)
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
    fn decode(d: &mut D) -> Result<Self, D::Error> {
        Fingerprint::decode(d).map(|f| PackedFingerprint(f))
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
