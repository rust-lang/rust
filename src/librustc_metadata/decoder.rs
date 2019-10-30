// Decoding metadata from a single crate's metadata

use crate::cstore::{CrateMetadata, MetadataBlob};
use crate::schema::*;
use crate::table::{FixedSizeEncoding, PerDefTable};

use rustc::hir::def_id::{CrateNum, DefId, DefIndex, LocalDefId, LOCAL_CRATE};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc::mir::{self, interpret};
use rustc::mir::interpret::AllocDecodingSession;
use rustc::session::Session;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::codec::TyDecoder;
use rustc::util::captures::Captures;

use std::io;
use std::mem;
use std::num::NonZeroUsize;
use std::u32;

use rustc_serialize::{Decodable, Decoder, Encodable, SpecializedDecoder, opaque};
use syntax::ast::Ident;
use syntax_pos::{self, Span, BytePos, DUMMY_SP};
use syntax_pos::symbol::Symbol;

crate struct DecodeContext<'a, 'tcx> {
    opaque: opaque::Decoder<'a>,
    cdata: Option<&'a CrateMetadata>,
    sess: Option<&'tcx Session>,
    tcx: Option<TyCtxt<'tcx>>,

    // Cache the last used source_file for translating spans as an optimization.
    last_source_file_index: usize,

    lazy_state: LazyState,

    // Used for decoding interpret::AllocIds in a cached & thread-safe manner.
    alloc_decoding_session: Option<AllocDecodingSession<'a>>,
}

/// Abstract over the various ways one can create metadata decoders.
crate trait Metadata<'a, 'tcx>: Copy {
    fn raw_bytes(self) -> &'a [u8];
    fn cdata(self) -> Option<&'a CrateMetadata> { None }
    fn sess(self) -> Option<&'tcx Session> { None }
    fn tcx(self) -> Option<TyCtxt<'tcx>> { None }

    fn decoder(self, pos: usize) -> DecodeContext<'a, 'tcx> {
        let tcx = self.tcx();
        DecodeContext {
            opaque: opaque::Decoder::new(self.raw_bytes(), pos),
            cdata: self.cdata(),
            sess: self.sess().or(tcx.map(|tcx| tcx.sess)),
            tcx,
            last_source_file_index: 0,
            lazy_state: LazyState::NoNode,
            alloc_decoding_session: self.cdata().map(|cdata| {
                cdata.alloc_decoding_state.new_decoding_session()
            }),
        }
    }
}

impl<'a, 'tcx> Metadata<'a, 'tcx> for &'a MetadataBlob {
    fn raw_bytes(self) -> &'a [u8] {
        &self.0
    }
}


impl<'a, 'tcx> Metadata<'a, 'tcx> for (&'a MetadataBlob, &'tcx Session) {
    fn raw_bytes(self) -> &'a [u8] {
        let (blob, _) = self;
        &blob.0
    }

    fn sess(self) -> Option<&'tcx Session> {
        let (_, sess) = self;
        Some(sess)
    }
}


impl<'a, 'tcx> Metadata<'a, 'tcx> for &'a CrateMetadata {
    fn raw_bytes(self) -> &'a [u8] {
        self.blob.raw_bytes()
    }
    fn cdata(self) -> Option<&'a CrateMetadata> {
        Some(self)
    }
}

impl<'a, 'tcx> Metadata<'a, 'tcx> for (&'a CrateMetadata, &'tcx Session) {
    fn raw_bytes(self) -> &'a [u8] {
        self.0.raw_bytes()
    }
    fn cdata(self) -> Option<&'a CrateMetadata> {
        Some(self.0)
    }
    fn sess(self) -> Option<&'tcx Session> {
        Some(&self.1)
    }
}

impl<'a, 'tcx> Metadata<'a, 'tcx> for (&'a CrateMetadata, TyCtxt<'tcx>) {
    fn raw_bytes(self) -> &'a [u8] {
        self.0.raw_bytes()
    }
    fn cdata(self) -> Option<&'a CrateMetadata> {
        Some(self.0)
    }
    fn tcx(self) -> Option<TyCtxt<'tcx>> {
        Some(self.1)
    }
}

impl<'a, 'tcx, T: Encodable + Decodable> Lazy<T> {
    crate fn decode<M: Metadata<'a, 'tcx>>(self, metadata: M) -> T {
        let mut dcx = metadata.decoder(self.position.get());
        dcx.lazy_state = LazyState::NodeStart(self.position);
        T::decode(&mut dcx).unwrap()
    }
}

impl<'a: 'x, 'tcx: 'x, 'x, T: Encodable + Decodable> Lazy<[T]> {
    crate fn decode<M: Metadata<'a, 'tcx>>(
        self,
        metadata: M,
    ) -> impl ExactSizeIterator<Item = T> + Captures<'a> + Captures<'tcx> + 'x {
        let mut dcx = metadata.decoder(self.position.get());
        dcx.lazy_state = LazyState::NodeStart(self.position);
        (0..self.meta).map(move |_| T::decode(&mut dcx).unwrap())
    }
}

impl<'a, 'tcx> DecodeContext<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx.expect("missing TyCtxt in DecodeContext")
    }

    fn cdata(&self) -> &'a CrateMetadata {
        self.cdata.expect("missing CrateMetadata in DecodeContext")
    }

    fn read_lazy_with_meta<T: ?Sized + LazyMeta>(
        &mut self,
        meta: T::Meta,
    ) -> Result<Lazy<T>, <Self as Decoder>::Error> {
        let min_size = T::min_size(meta);
        let distance = self.read_usize()?;
        let position = match self.lazy_state {
            LazyState::NoNode => bug!("read_lazy_with_meta: outside of a metadata node"),
            LazyState::NodeStart(start) => {
                let start = start.get();
                assert!(distance + min_size <= start);
                start - distance - min_size
            }
            LazyState::Previous(last_min_end) => last_min_end.get() + distance,
        };
        self.lazy_state = LazyState::Previous(NonZeroUsize::new(position + min_size).unwrap());
        Ok(Lazy::from_position_and_meta(NonZeroUsize::new(position).unwrap(), meta))
    }
}

impl<'a, 'tcx> TyDecoder<'tcx> for DecodeContext<'a, 'tcx> {
    #[inline]
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx.expect("missing TyCtxt in DecodeContext")
    }

    #[inline]
    fn peek_byte(&self) -> u8 {
        self.opaque.data[self.opaque.position()]
    }

    #[inline]
    fn position(&self) -> usize {
        self.opaque.position()
    }

    fn cached_ty_for_shorthand<F>(&mut self,
                                  shorthand: usize,
                                  or_insert_with: F)
                                  -> Result<Ty<'tcx>, Self::Error>
        where F: FnOnce(&mut Self) -> Result<Ty<'tcx>, Self::Error>
    {
        let tcx = self.tcx();

        let key = ty::CReaderCacheKey {
            cnum: self.cdata().cnum,
            pos: shorthand,
        };

        if let Some(&ty) = tcx.rcache.borrow().get(&key) {
            return Ok(ty);
        }

        let ty = or_insert_with(self)?;
        tcx.rcache.borrow_mut().insert(key, ty);
        Ok(ty)
    }

    fn with_position<F, R>(&mut self, pos: usize, f: F) -> R
        where F: FnOnce(&mut Self) -> R
    {
        let new_opaque = opaque::Decoder::new(self.opaque.data, pos);
        let old_opaque = mem::replace(&mut self.opaque, new_opaque);
        let old_state = mem::replace(&mut self.lazy_state, LazyState::NoNode);
        let r = f(self);
        self.opaque = old_opaque;
        self.lazy_state = old_state;
        r
    }

    fn map_encoded_cnum_to_current(&self, cnum: CrateNum) -> CrateNum {
        if cnum == LOCAL_CRATE {
            self.cdata().cnum
        } else {
            self.cdata().cnum_map[cnum]
        }
    }
}

impl<'a, 'tcx, T: Encodable> SpecializedDecoder<Lazy<T>> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<Lazy<T>, Self::Error> {
        self.read_lazy_with_meta(())
    }
}

impl<'a, 'tcx, T: Encodable> SpecializedDecoder<Lazy<[T]>> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<Lazy<[T]>, Self::Error> {
        let len = self.read_usize()?;
        if len == 0 {
            Ok(Lazy::empty())
        } else {
            self.read_lazy_with_meta(len)
        }
    }
}

impl<'a, 'tcx, T> SpecializedDecoder<Lazy<PerDefTable<T>>> for DecodeContext<'a, 'tcx>
    where Option<T>: FixedSizeEncoding,
{
    fn specialized_decode(&mut self) -> Result<Lazy<PerDefTable<T>>, Self::Error> {
        let len = self.read_usize()?;
        self.read_lazy_with_meta(len)
    }
}

impl<'a, 'tcx> SpecializedDecoder<DefId> for DecodeContext<'a, 'tcx> {
    #[inline]
    fn specialized_decode(&mut self) -> Result<DefId, Self::Error> {
        let krate = CrateNum::decode(self)?;
        let index = DefIndex::decode(self)?;

        Ok(DefId {
            krate,
            index,
        })
    }
}

impl<'a, 'tcx> SpecializedDecoder<DefIndex> for DecodeContext<'a, 'tcx> {
    #[inline]
    fn specialized_decode(&mut self) -> Result<DefIndex, Self::Error> {
        Ok(DefIndex::from_u32(self.read_u32()?))
    }
}

impl<'a, 'tcx> SpecializedDecoder<LocalDefId> for DecodeContext<'a, 'tcx> {
    #[inline]
    fn specialized_decode(&mut self) -> Result<LocalDefId, Self::Error> {
        self.specialized_decode().map(|i| LocalDefId::from_def_id(i))
    }
}

impl<'a, 'tcx> SpecializedDecoder<interpret::AllocId> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<interpret::AllocId, Self::Error> {
        if let Some(alloc_decoding_session) = self.alloc_decoding_session {
            alloc_decoding_session.decode_alloc_id(self)
        } else {
            bug!("Attempting to decode interpret::AllocId without CrateMetadata")
        }
    }
}

impl<'a, 'tcx> SpecializedDecoder<Span> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<Span, Self::Error> {
        let tag = u8::decode(self)?;

        if tag == TAG_INVALID_SPAN {
            return Ok(DUMMY_SP)
        }

        debug_assert_eq!(tag, TAG_VALID_SPAN);

        let lo = BytePos::decode(self)?;
        let len = BytePos::decode(self)?;
        let hi = lo + len;

        let sess = if let Some(sess) = self.sess {
            sess
        } else {
            bug!("Cannot decode Span without Session.")
        };

        let imported_source_files = self.cdata().imported_source_files(&sess.source_map());
        let source_file = {
            // Optimize for the case that most spans within a translated item
            // originate from the same source_file.
            let last_source_file = &imported_source_files[self.last_source_file_index];

            if lo >= last_source_file.original_start_pos &&
               lo <= last_source_file.original_end_pos {
                last_source_file
            } else {
                let mut a = 0;
                let mut b = imported_source_files.len();

                while b - a > 1 {
                    let m = (a + b) / 2;
                    if imported_source_files[m].original_start_pos > lo {
                        b = m;
                    } else {
                        a = m;
                    }
                }

                self.last_source_file_index = a;
                &imported_source_files[a]
            }
        };

        // Make sure our binary search above is correct.
        debug_assert!(lo >= source_file.original_start_pos &&
                      lo <= source_file.original_end_pos);

        // Make sure we correctly filtered out invalid spans during encoding
        debug_assert!(hi >= source_file.original_start_pos &&
                      hi <= source_file.original_end_pos);

        let lo = (lo + source_file.translated_source_file.start_pos)
                 - source_file.original_start_pos;
        let hi = (hi + source_file.translated_source_file.start_pos)
                 - source_file.original_start_pos;

        Ok(Span::with_root_ctxt(lo, hi))
    }
}

impl SpecializedDecoder<Ident> for DecodeContext<'_, '_> {
    fn specialized_decode(&mut self) -> Result<Ident, Self::Error> {
        // FIXME(jseyfried): intercrate hygiene

        Ok(Ident::with_dummy_span(Symbol::decode(self)?))
    }
}

impl<'a, 'tcx> SpecializedDecoder<Fingerprint> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<Fingerprint, Self::Error> {
        Fingerprint::decode_opaque(&mut self.opaque)
    }
}

impl<'a, 'tcx, T: Decodable> SpecializedDecoder<mir::ClearCrossCrate<T>>
for DecodeContext<'a, 'tcx> {
    #[inline]
    fn specialized_decode(&mut self) -> Result<mir::ClearCrossCrate<T>, Self::Error> {
        Ok(mir::ClearCrossCrate::Clear)
    }
}

implement_ty_decoder!( DecodeContext<'a, 'tcx> );

impl<'tcx> MetadataBlob {
    crate fn is_compatible(&self) -> bool {
        self.raw_bytes().starts_with(METADATA_HEADER)
    }

    crate fn get_rustc_version(&self) -> String {
        Lazy::<String>::from_position(
            NonZeroUsize::new(METADATA_HEADER.len() + 4).unwrap(),
        ).decode(self)
    }

    crate fn get_root(&self) -> CrateRoot<'tcx> {
        let slice = self.raw_bytes();
        let offset = METADATA_HEADER.len();
        let pos = (((slice[offset + 0] as u32) << 24) | ((slice[offset + 1] as u32) << 16) |
                   ((slice[offset + 2] as u32) << 8) |
                   ((slice[offset + 3] as u32) << 0)) as usize;
        Lazy::<CrateRoot<'tcx>>::from_position(
            NonZeroUsize::new(pos).unwrap(),
        ).decode(self)
    }

    crate fn list_crate_metadata(&self,
                               out: &mut dyn io::Write) -> io::Result<()> {
        write!(out, "=External Dependencies=\n")?;
        let root = self.get_root();
        for (i, dep) in root.crate_deps
                            .decode(self)
                            .enumerate() {
            write!(out, "{} {}{}\n", i + 1, dep.name, dep.extra_filename)?;
        }
        write!(out, "\n")?;
        Ok(())
    }
}
