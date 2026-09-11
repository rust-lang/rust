use rustc_data_structures::fx::FxHashMap;
use rustc_hashes::Hash64;
use rustc_hir::def_path_hash_map::DefPathHashMap;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use rustc_span::def_id::{DefIndex, DefPathHash};

use crate::rmeta::EncodeContext;
use crate::rmeta::decoder::BlobDecodeContext;

pub(crate) enum DefPathHashMapRef<'tcx> {
    OwnedFromMetadata(FxHashMap<Hash64, DefIndex>),
    BorrowedFromTcx(&'tcx DefPathHashMap),
}

impl DefPathHashMapRef<'_> {
    #[inline]
    pub(crate) fn def_path_hash_to_def_index(
        &self,
        def_path_hash: &DefPathHash,
    ) -> Option<DefIndex> {
        match *self {
            DefPathHashMapRef::OwnedFromMetadata(ref map) => {
                map.get(&def_path_hash.local_hash()).copied()
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
            DefPathHashMapRef::BorrowedFromTcx(def_path_hash_map) => {
                e.emit_usize(def_path_hash_map.len());

                for (h, index) in def_path_hash_map.iter() {
                    h.encode(e);
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

        let mut map = FxHashMap::default();

        for _ in 0..len {
            map.insert(Hash64::decode(d), DefIndex::decode(d));
        }

        DefPathHashMapRef::OwnedFromMetadata(map)
    }
}
