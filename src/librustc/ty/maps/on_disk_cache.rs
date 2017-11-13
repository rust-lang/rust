// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use dep_graph::{DepNodeIndex, SerializedDepNodeIndex};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::Idx;
use errors::Diagnostic;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder, opaque,
                      SpecializedDecoder, SpecializedEncoder};
use session::Session;
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::mem;
use syntax::codemap::{CodeMap, StableFilemapId};
use syntax_pos::{BytePos, Span, NO_EXPANSION, DUMMY_SP};
use ty;
use ty::codec::{self as ty_codec};
use ty::context::TyCtxt;

/// `OnDiskCache` provides an interface to incr. comp. data cached from the
/// previous compilation session. This data will eventually include the results
/// of a few selected queries (like `typeck_tables_of` and `mir_optimized`) and
/// any diagnostics that have been emitted during a query.
pub struct OnDiskCache<'sess> {
    // The diagnostics emitted during the previous compilation session.
    prev_diagnostics: FxHashMap<SerializedDepNodeIndex, Vec<Diagnostic>>,

    // This field collects all Diagnostics emitted during the current
    // compilation session.
    current_diagnostics: RefCell<FxHashMap<DepNodeIndex, Vec<Diagnostic>>>,

    // This will eventually be needed for creating Decoders that can rebase
    // spans.
    _prev_filemap_starts: BTreeMap<BytePos, StableFilemapId>,
    codemap: &'sess CodeMap,
}

// This type is used only for (de-)serialization.
#[derive(RustcEncodable, RustcDecodable)]
struct Header {
    prev_filemap_starts: BTreeMap<BytePos, StableFilemapId>,
}

type EncodedPrevDiagnostics = Vec<(SerializedDepNodeIndex, Vec<Diagnostic>)>;

impl<'sess> OnDiskCache<'sess> {
    /// Create a new OnDiskCache instance from the serialized data in `data`.
    /// Note that the current implementation (which only deals with diagnostics
    /// so far) will eagerly deserialize the complete cache. Once we are
    /// dealing with larger amounts of data (i.e. cached query results),
    /// deserialization will need to happen lazily.
    pub fn new(sess: &'sess Session, data: &[u8], start_pos: usize) -> OnDiskCache<'sess> {
        debug_assert!(sess.opts.incremental.is_some());

        let mut decoder = opaque::Decoder::new(&data[..], start_pos);
        let header = Header::decode(&mut decoder).unwrap();

        let prev_diagnostics = {
            let mut decoder = CacheDecoder {
                opaque: decoder,
                codemap: sess.codemap(),
                prev_filemap_starts: &header.prev_filemap_starts,
            };

            let prev_diagnostics: FxHashMap<_, _> = {
                let diagnostics = EncodedPrevDiagnostics::decode(&mut decoder)
                    .expect("Error while trying to decode prev. diagnostics \
                             from incr. comp. cache.");
                diagnostics.into_iter().collect()
            };

            prev_diagnostics
        };

        OnDiskCache {
            prev_diagnostics,
            _prev_filemap_starts: header.prev_filemap_starts,
            codemap: sess.codemap(),
            current_diagnostics: RefCell::new(FxHashMap()),
        }
    }

    pub fn new_empty(codemap: &'sess CodeMap) -> OnDiskCache<'sess> {
        OnDiskCache {
            prev_diagnostics: FxHashMap(),
            _prev_filemap_starts: BTreeMap::new(),
            codemap,
            current_diagnostics: RefCell::new(FxHashMap()),
        }
    }

    pub fn serialize<'a, 'gcx, 'lcx, E>(&self,
                                        tcx: TyCtxt<'a, 'gcx, 'lcx>,
                                        encoder: &mut E)
                                        -> Result<(), E::Error>
        where E: ty_codec::TyEncoder
     {
        // Serializing the DepGraph should not modify it:
        let _in_ignore = tcx.dep_graph.in_ignore();

        let mut encoder = CacheEncoder {
            encoder,
            type_shorthands: FxHashMap(),
            predicate_shorthands: FxHashMap(),
        };

        let prev_filemap_starts: BTreeMap<_, _> = self
            .codemap
            .files()
            .iter()
            .map(|fm| (fm.start_pos, StableFilemapId::new(fm)))
            .collect();

        Header { prev_filemap_starts }.encode(&mut encoder)?;

        let diagnostics: EncodedPrevDiagnostics =
            self.current_diagnostics
                .borrow()
                .iter()
                .map(|(k, v)| (SerializedDepNodeIndex::new(k.index()), v.clone()))
                .collect();

        diagnostics.encode(&mut encoder)?;

        Ok(())
    }

    /// Load a diagnostic emitted during the previous compilation session.
    pub fn load_diagnostics(&self,
                            dep_node_index: SerializedDepNodeIndex)
                            -> Vec<Diagnostic> {
        self.prev_diagnostics.get(&dep_node_index).cloned().unwrap_or(vec![])
    }

    /// Store a diagnostic emitted during the current compilation session.
    /// Anything stored like this will be available via `load_diagnostics` in
    /// the next compilation session.
    pub fn store_diagnostics(&self,
                             dep_node_index: DepNodeIndex,
                             diagnostics: Vec<Diagnostic>) {
        let mut current_diagnostics = self.current_diagnostics.borrow_mut();
        let prev = current_diagnostics.insert(dep_node_index, diagnostics);
        debug_assert!(prev.is_none());
    }

    /// Store a diagnostic emitted during computation of an anonymous query.
    /// Since many anonymous queries can share the same `DepNode`, we aggregate
    /// them -- as opposed to regular queries where we assume that there is a
    /// 1:1 relationship between query-key and `DepNode`.
    pub fn store_diagnostics_for_anon_node(&self,
                                           dep_node_index: DepNodeIndex,
                                           mut diagnostics: Vec<Diagnostic>) {
        let mut current_diagnostics = self.current_diagnostics.borrow_mut();

        let x = current_diagnostics.entry(dep_node_index).or_insert_with(|| {
            mem::replace(&mut diagnostics, Vec::new())
        });

        x.extend(diagnostics.into_iter());
    }
}


//- DECODING -------------------------------------------------------------------

/// A decoder that can read the incr. comp. cache. It is similar to the one
/// we use for crate metadata decoding in that it can rebase spans and
/// eventually will also handle things that contain `Ty` instances.
struct CacheDecoder<'a> {
    opaque: opaque::Decoder<'a>,
    codemap: &'a CodeMap,
    prev_filemap_starts: &'a BTreeMap<BytePos, StableFilemapId>,
}

impl<'a> CacheDecoder<'a> {
    fn find_filemap_prev_bytepos(&self,
                                 prev_bytepos: BytePos)
                                 -> Option<(BytePos, StableFilemapId)> {
        for (start, id) in self.prev_filemap_starts.range(BytePos(0) ..= prev_bytepos).rev() {
            return Some((*start, *id))
        }

        None
    }
}

macro_rules! decoder_methods {
    ($($name:ident -> $ty:ty;)*) => {
        $(fn $name(&mut self) -> Result<$ty, Self::Error> {
            self.opaque.$name()
        })*
    }
}

impl<'sess> Decoder for CacheDecoder<'sess> {
    type Error = String;

    decoder_methods! {
        read_nil -> ();

        read_u128 -> u128;
        read_u64 -> u64;
        read_u32 -> u32;
        read_u16 -> u16;
        read_u8 -> u8;
        read_usize -> usize;

        read_i128 -> i128;
        read_i64 -> i64;
        read_i32 -> i32;
        read_i16 -> i16;
        read_i8 -> i8;
        read_isize -> isize;

        read_bool -> bool;
        read_f64 -> f64;
        read_f32 -> f32;
        read_char -> char;
        read_str -> Cow<str>;
    }

    fn error(&mut self, err: &str) -> Self::Error {
        self.opaque.error(err)
    }
}

impl<'a> SpecializedDecoder<Span> for CacheDecoder<'a> {
    fn specialized_decode(&mut self) -> Result<Span, Self::Error> {
        let lo = BytePos::decode(self)?;
        let hi = BytePos::decode(self)?;

        if let Some((prev_filemap_start, filemap_id)) = self.find_filemap_prev_bytepos(lo) {
            if let Some(current_filemap) = self.codemap.filemap_by_stable_id(filemap_id) {
                let lo = (lo + current_filemap.start_pos) - prev_filemap_start;
                let hi = (hi + current_filemap.start_pos) - prev_filemap_start;
                return Ok(Span::new(lo, hi, NO_EXPANSION));
            }
        }

        Ok(DUMMY_SP)
    }
}


//- ENCODING -------------------------------------------------------------------

struct CacheEncoder<'enc, 'tcx, E>
    where E: 'enc + ty_codec::TyEncoder
{
    encoder: &'enc mut E,
    type_shorthands: FxHashMap<ty::Ty<'tcx>, usize>,
    predicate_shorthands: FxHashMap<ty::Predicate<'tcx>, usize>,
}

impl<'enc, 'tcx, E> ty_codec::TyEncoder for CacheEncoder<'enc, 'tcx, E>
    where E: 'enc + ty_codec::TyEncoder
{
    fn position(&self) -> usize {
        self.encoder.position()
    }
}

impl<'enc, 'tcx, E> SpecializedEncoder<ty::Ty<'tcx>> for CacheEncoder<'enc, 'tcx, E>
    where E: 'enc + ty_codec::TyEncoder
{
    fn specialized_encode(&mut self, ty: &ty::Ty<'tcx>) -> Result<(), Self::Error> {
        ty_codec::encode_with_shorthand(self, ty,
            |encoder| &mut encoder.type_shorthands)
    }
}

impl<'enc, 'tcx, E> SpecializedEncoder<ty::GenericPredicates<'tcx>>
    for CacheEncoder<'enc, 'tcx, E>
    where E: 'enc + ty_codec::TyEncoder
{
    fn specialized_encode(&mut self,
                          predicates: &ty::GenericPredicates<'tcx>)
                          -> Result<(), Self::Error> {
        ty_codec::encode_predicates(self, predicates,
            |encoder| &mut encoder.predicate_shorthands)
    }
}

macro_rules! encoder_methods {
    ($($name:ident($ty:ty);)*) => {
        $(fn $name(&mut self, value: $ty) -> Result<(), Self::Error> {
            self.encoder.$name(value)
        })*
    }
}

impl<'enc, 'tcx, E> Encoder for CacheEncoder<'enc, 'tcx, E>
    where E: 'enc + ty_codec::TyEncoder
{
    type Error = E::Error;

    fn emit_nil(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    encoder_methods! {
        emit_usize(usize);
        emit_u128(u128);
        emit_u64(u64);
        emit_u32(u32);
        emit_u16(u16);
        emit_u8(u8);

        emit_isize(isize);
        emit_i128(i128);
        emit_i64(i64);
        emit_i32(i32);
        emit_i16(i16);
        emit_i8(i8);

        emit_bool(bool);
        emit_f64(f64);
        emit_f32(f32);
        emit_char(char);
        emit_str(&str);
    }
}
