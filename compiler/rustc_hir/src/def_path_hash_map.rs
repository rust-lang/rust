use rustc_data_structures::fingerprint::Fingerprint;
use rustc_span::def_id::{DefIndex, DefPathHash};

#[derive(Clone, Default)]
pub struct Config;

impl odht::Config for Config {
    type Key = DefPathHash;
    type Value = DefIndex;

    type EncodedKey = [u8; 16];
    type EncodedValue = [u8; 4];

    type H = odht::UnHashFn;

    #[inline]
    fn encode_key(k: &DefPathHash) -> [u8; 16] {
        k.0.to_le_bytes()
    }

    #[inline]
    fn encode_value(v: &DefIndex) -> [u8; 4] {
        v.as_u32().to_le_bytes()
    }

    #[inline]
    fn decode_key(k: &[u8; 16]) -> DefPathHash {
        DefPathHash(Fingerprint::from_le_bytes(*k))
    }

    #[inline]
    fn decode_value(v: &[u8; 4]) -> DefIndex {
        DefIndex::from_u32(u32::from_le_bytes(*v))
    }
}

pub type DefPathHashMap = odht::HashTableOwned<Config>;
