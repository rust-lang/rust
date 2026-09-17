use rustc_hashes::Hash64;
use rustc_hir::def_path_hash_map::Config as HashMapConfig;
use rustc_hir::definitions::DefPathToIndexMap;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use rustc_span::def_id::{DefIndex, DefPathHash};

use crate::rmeta::EncodeContext;
use crate::rmeta::decoder::BlobDecodeContext;

pub(crate) enum DefPathHashMapRef<'tcx> {
    OwnedFromMetadata(odht::HashTableOwned<HashMapConfig>),
    BorrowedFromTcx(&'tcx DefPathToIndexMap),
}

impl DefPathHashMapRef<'_> {
    #[inline]
    pub(crate) fn def_path_hash_to_def_index(
        &self,
        def_path_hash: &DefPathHash,
    ) -> Option<DefIndex> {
        match *self {
            DefPathHashMapRef::OwnedFromMetadata(ref map) => map.get(&def_path_hash.local_hash()),
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
                #[allow(rustc::potential_query_instability)]
                let bytes = map.det_part.raw_bytes();
                e.emit_usize(bytes.len());
                e.emit_raw_bytes(bytes);

                #[allow(rustc::potential_query_instability)]
                let mut vec = map.non_det_part.iter().collect::<Vec<_>>();
                vec.sort_by_key(|(hash, _)| *hash);

                e.emit_usize(vec.len());
                for (hash, index) in vec {
                    hash.encode(e);
                    index.encode(e);
                }
            }
            DefPathHashMapRef::OwnedFromMetadata(_) => {
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

        let mut inner = odht::HashTableOwned::from_raw_bytes(o.as_ref()).unwrap_or_else(|e| {
            panic!("decode error: {e}");
        });

        let non_det_size = d.read_usize();
        for _ in 0..non_det_size {
            inner.insert(&Hash64::decode(d), &DefIndex::decode(d));
        }

        DefPathHashMapRef::OwnedFromMetadata(inner)
    }
}
