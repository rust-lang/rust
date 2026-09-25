use rustc_data_structures::owned_slice::OwnedSlice;
use rustc_data_structures::sorted_map::SortedMap;
use rustc_hashes::Hash64;
use rustc_hir::def_path_hash_map::Config as HashMapConfig;
use rustc_hir::definitions::DefPathToIndexMap;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use rustc_span::def_id::{DefIndex, DefPathHash};

use crate::rmeta::EncodeContext;
use crate::rmeta::decoder::BlobDecodeContext;

pub(crate) enum DefPathHashMapRef<'tcx> {
    OwnedFromMetadata(odht::HashTable<HashMapConfig, OwnedSlice>, SortedMap<Hash64, DefIndex>),
    BorrowedFromTcx(&'tcx DefPathToIndexMap),
}

impl DefPathHashMapRef<'_> {
    #[inline]
    pub(crate) fn def_path_hash_to_def_index(
        &self,
        def_path_hash: &DefPathHash,
    ) -> Option<DefIndex> {
        match self {
            DefPathHashMapRef::OwnedFromMetadata(det_map, non_det_map) => {
                let hash = &def_path_hash.local_hash();
                det_map.get(hash).or_else(|| non_det_map.get(hash).copied())
            }
            DefPathHashMapRef::BorrowedFromTcx(_) => {
                panic!("DefPathHashMap::BorrowedFromTcx variant only exists for serialization")
            }
        }
    }
}

impl<'a, 'tcx> Encodable<EncodeContext<'a, 'tcx>> for DefPathHashMapRef<'tcx> {
    fn encode(&self, e: &mut EncodeContext<'a, 'tcx>) {
        match *self {
            DefPathHashMapRef::BorrowedFromTcx(map) => {
                let bytes = map.det_part.raw_bytes();
                e.emit_usize(bytes.len());
                e.emit_raw_bytes(bytes);

                map.non_det_part.as_ref().unwrap_or(&Default::default()).range(..).encode(e);
            }
            DefPathHashMapRef::OwnedFromMetadata(..) => {
                panic!("DefPathHashMap::OwnedFromMetadata variant only exists for deserialization")
            }
        }
    }
}

impl<'a> Decodable<BlobDecodeContext<'a>> for DefPathHashMapRef<'static> {
    fn decode(d: &mut BlobDecodeContext<'a>) -> DefPathHashMapRef<'static> {
        let len = d.read_usize();
        let pos = d.position();
        let o = d.blob().bytes().clone().slice(|blob| &blob[pos..pos + len]);

        // Although we already have the data we need via the `OwnedSlice`, we still need
        // to advance the `DecodeContext`'s position so it's in a valid state after
        // the method. We use `read_raw_bytes()` for that.
        let _ = d.read_raw_bytes(len);

        let inner = odht::HashTable::from_raw_bytes(o).unwrap_or_else(|e| {
            panic!("decode error: {e}");
        });

        let elements = Vec::<(Hash64, DefIndex)>::decode(d);
        let non_det_map = SortedMap::from_presorted_elements(elements);

        DefPathHashMapRef::OwnedFromMetadata(inner, non_det_map)
    }
}
