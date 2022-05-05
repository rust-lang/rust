use crate::rmeta::DecodeContext;
use crate::rmeta::EncodeContext;
use crate::rmeta::MetadataBlob;
use rustc_data_structures::owning_ref::OwningRef;
use rustc_hir::def_path_hash_map::{Config as HashMapConfig, DefPathHashMap};
use rustc_serialize::{opaque, Decodable, Decoder, Encodable, Encoder};
use rustc_span::def_id::{DefIndex, DefPathHash};

crate enum DefPathHashMapRef<'tcx> {
    OwnedFromMetadata(odht::HashTable<HashMapConfig, OwningRef<MetadataBlob, [u8]>>),
    BorrowedFromTcx(&'tcx DefPathHashMap),
}

impl DefPathHashMapRef<'_> {
    #[inline]
    pub fn def_path_hash_to_def_index(&self, def_path_hash: &DefPathHash) -> DefIndex {
        match *self {
            DefPathHashMapRef::OwnedFromMetadata(ref map) => map.get(def_path_hash).unwrap(),
            DefPathHashMapRef::BorrowedFromTcx(_) => {
                panic!("DefPathHashMap::BorrowedFromTcx variant only exists for serialization")
            }
        }
    }
}

impl<'a, 'tcx> Encodable<EncodeContext<'a, 'tcx>> for DefPathHashMapRef<'tcx> {
    fn encode(&self, e: &mut EncodeContext<'a, 'tcx>) -> opaque::EncodeResult {
        match *self {
            DefPathHashMapRef::BorrowedFromTcx(def_path_hash_map) => {
                let bytes = def_path_hash_map.raw_bytes();
                e.emit_usize(bytes.len())?;
                e.emit_raw_bytes(bytes)
            }
            DefPathHashMapRef::OwnedFromMetadata(_) => {
                panic!("DefPathHashMap::OwnedFromMetadata variant only exists for deserialization")
            }
        }
    }
}

impl<'a, 'tcx> Decodable<DecodeContext<'a, 'tcx>> for DefPathHashMapRef<'static> {
    fn decode(d: &mut DecodeContext<'a, 'tcx>) -> DefPathHashMapRef<'static> {
        // Import TyDecoder so we can access the DecodeContext::position() method
        use crate::rustc_middle::ty::codec::TyDecoder;

        let len = d.read_usize();
        let pos = d.position();
        let o = OwningRef::new(d.blob().clone()).map(|x| &x[pos..pos + len]);

        // Although we already have the data we need via the OwningRef, we still need
        // to advance the DecodeContext's position so it's in a valid state after
        // the method. We use read_raw_bytes() for that.
        let _ = d.read_raw_bytes(len);

        let inner = odht::HashTable::from_raw_bytes(o).unwrap_or_else(|e| {
            panic!("decode error: {}", e);
        });
        DefPathHashMapRef::OwnedFromMetadata(inner)
    }
}
