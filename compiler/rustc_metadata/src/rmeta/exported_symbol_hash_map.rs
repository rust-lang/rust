use rustc_data_structures::owned_slice::OwnedSlice;
use rustc_hashes::Hash128;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};

use crate::rmeta::EncodeContext;
use crate::rmeta::decoder::BlobDecodeContext;

#[derive(Clone, Default)]
pub(crate) struct ExportedSymbolHashConfig;

impl odht::Config for ExportedSymbolHashConfig {
    type Key = Hash128;
    type Value = ();

    type EncodedKey = [u8; 16];
    type EncodedValue = [u8; 0];

    type H = odht::UnHashFn;

    #[inline]
    fn encode_key(k: &Hash128) -> [u8; 16] {
        k.as_u128().to_le_bytes()
    }

    #[inline]
    fn encode_value(_v: &()) -> [u8; 0] {
        []
    }

    #[inline]
    fn decode_key(k: &[u8; 16]) -> Hash128 {
        Hash128::new(u128::from_le_bytes(*k))
    }

    #[inline]
    fn decode_value(_v: &[u8; 0]) {}
}

pub(crate) type ExportedSymbolHashTable = odht::HashTableOwned<ExportedSymbolHashConfig>;

pub(crate) enum ExportedSymbolHashTableRef {
    /// Zero-copy view into metadata bytes.
    OwnedFromMetadata(odht::HashTable<ExportedSymbolHashConfig, OwnedSlice>),
    /// Locally built table, used during encoding.
    Owned(ExportedSymbolHashTable),
}

impl ExportedSymbolHashTableRef {
    pub(crate) fn keys(&self) -> Vec<Hash128> {
        match self {
            Self::OwnedFromMetadata(table) => table.iter().map(|(k, ())| k).collect(),
            Self::Owned(table) => table.iter().map(|(k, ())| k).collect(),
        }
    }
}

impl<'a, 'tcx> Encodable<EncodeContext<'a, 'tcx>> for ExportedSymbolHashTableRef {
    fn encode(&self, e: &mut EncodeContext<'a, 'tcx>) {
        let raw_bytes = match self {
            ExportedSymbolHashTableRef::Owned(table) => table.raw_bytes(),
            ExportedSymbolHashTableRef::OwnedFromMetadata(_) => {
                panic!(
                    "ExportedSymbolHashTableRef::OwnedFromMetadata variant \
                     only exists for deserialization"
                )
            }
        };
        e.emit_usize(raw_bytes.len());
        e.emit_raw_bytes(raw_bytes);
    }
}

impl<'a> Decodable<BlobDecodeContext<'a>> for ExportedSymbolHashTableRef {
    fn decode(d: &mut BlobDecodeContext<'a>) -> ExportedSymbolHashTableRef {
        let len = d.read_usize();
        let pos = d.position();
        let o = d.blob().bytes().clone().slice(|blob| &blob[pos..pos + len]);

        // Advance the decoder's position past the raw bytes we just sliced.
        let _ = d.read_raw_bytes(len);

        let inner = odht::HashTable::from_raw_bytes(o).unwrap_or_else(|e| {
            panic!("decode error for ExportedSymbolHashTable: {e}");
        });
        ExportedSymbolHashTableRef::OwnedFromMetadata(inner)
    }
}
