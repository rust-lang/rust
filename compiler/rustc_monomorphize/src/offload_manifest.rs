//! Offload manifest: communicates required generic kernel instantiations
//! between host-metadata and device compilation passes.
//!
//! Uses `TyEncoder`/`TyDecoder` to serialize `ty::Instance`. DefIds are
//! encoded as (crate name, DefPath) pairs for stability.

use std::fs;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lock;
use rustc_hir::def_id::{DefId, DefIndex, LOCAL_CRATE};
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mono::MonoItem;
use rustc_middle::ty::codec::{TyDecoder, TyEncoder};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_serialize::opaque::{FileEncoder, MemDecoder};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use rustc_span::{
    BlobDecoder, BytePos, ByteSymbol, Pos, Span, SpanDecoder, SpanEncoder, Symbol, SyntaxContext,
};

pub(crate) struct OffloadManifestEncoder<'a, 'tcx> {
    encoder: FileEncoder<'a>,
    type_shorthands: FxHashMap<Ty<'tcx>, usize>,
    predicate_shorthands: FxHashMap<ty::PredicateKind<'tcx>, usize>,
    tcx: TyCtxt<'tcx>,
}

impl<'a, 'tcx> OffloadManifestEncoder<'a, 'tcx> {
    pub(crate) fn new(path: &'a std::path::Path, tcx: TyCtxt<'tcx>) -> std::io::Result<Self> {
        let encoder = FileEncoder::new(path)?;
        Ok(OffloadManifestEncoder {
            encoder,
            type_shorthands: FxHashMap::default(),
            predicate_shorthands: FxHashMap::default(),
            tcx,
        })
    }

    pub(crate) fn finish(mut self) -> std::io::Result<()> {
        self.encoder.finish().map(|_| ()).map_err(|(_, e)| e)
    }
}

impl<'a, 'tcx> Encoder for OffloadManifestEncoder<'a, 'tcx> {
    fn emit_usize(&mut self, v: usize) {
        self.encoder.emit_usize(v);
    }
    fn emit_u128(&mut self, v: u128) {
        self.encoder.emit_u128(v);
    }
    fn emit_u64(&mut self, v: u64) {
        self.encoder.emit_u64(v);
    }
    fn emit_u32(&mut self, v: u32) {
        self.encoder.emit_u32(v);
    }
    fn emit_u16(&mut self, v: u16) {
        self.encoder.emit_u16(v);
    }
    fn emit_u8(&mut self, v: u8) {
        self.encoder.emit_u8(v);
    }
    fn emit_isize(&mut self, v: isize) {
        self.encoder.emit_isize(v);
    }
    fn emit_i128(&mut self, v: i128) {
        self.encoder.emit_i128(v);
    }
    fn emit_i64(&mut self, v: i64) {
        self.encoder.emit_i64(v);
    }
    fn emit_i32(&mut self, v: i32) {
        self.encoder.emit_i32(v);
    }
    fn emit_i16(&mut self, v: i16) {
        self.encoder.emit_i16(v);
    }
    fn emit_i8(&mut self, v: i8) {
        self.encoder.emit_i8(v);
    }
    fn emit_raw_bytes(&mut self, v: &[u8]) {
        self.encoder.emit_raw_bytes(v);
    }
}

impl<'a, 'tcx> SpanEncoder for OffloadManifestEncoder<'a, 'tcx> {
    fn encode_span(&mut self, _span: Span) {
        // Spans are not needed in the manifest, encode a dummy span.
        self.emit_usize(0);
        self.emit_usize(0);
        self.emit_u32(0);
    }

    fn encode_symbol(&mut self, sym: rustc_span::Symbol) {
        sym.as_str().encode(self);
    }

    fn encode_byte_symbol(&mut self, byte_sym: ByteSymbol) {
        let bytes = byte_sym.as_byte_str();
        debug_assert!(
            bytes.is_empty(),
            "ByteSymbols with content are not expected in offload manifests"
        );
        self.emit_usize(bytes.len());
        self.emit_raw_bytes(bytes.as_ref())
    }

    fn encode_expn_id(&mut self, _expn_id: rustc_span::ExpnId) {
        self.emit_u32(0);
    }

    fn encode_syntax_context(&mut self, _syntax_context: SyntaxContext) {
        self.emit_u32(0);
    }

    fn encode_crate_num(&mut self, crate_num: rustc_span::def_id::CrateNum) {
        crate_num.as_u32().encode(self);
    }

    fn encode_def_index(&mut self, def_index: rustc_span::def_id::DefIndex) {
        def_index.as_u32().encode(self);
    }

    fn encode_def_id(&mut self, def_id: rustc_span::def_id::DefId) {
        let crate_name = self.tcx.crate_name(def_id.krate);
        let def_path = self.tcx.def_path(def_id);
        crate_name.encode(self);
        def_path.to_string_no_crate_verbose().encode(self);
    }
}

impl<'a, 'tcx> TyEncoder<'tcx> for OffloadManifestEncoder<'a, 'tcx> {
    const CLEAR_CROSS_CRATE: bool = true;

    fn position(&self) -> usize {
        self.encoder.position()
    }

    fn type_shorthands(&mut self) -> &mut FxHashMap<Ty<'tcx>, usize> {
        &mut self.type_shorthands
    }

    fn predicate_shorthands(&mut self) -> &mut FxHashMap<ty::PredicateKind<'tcx>, usize> {
        &mut self.predicate_shorthands
    }

    fn encode_alloc_id(&mut self, _alloc_id: &rustc_middle::mir::interpret::AllocId) {
        // AllocIds are not expected in the manifest.
    }
}

const UNRESOLVED_DEF_ID: DefId = DefId {
    krate: rustc_span::def_id::CrateNum::MAX,
    index: rustc_span::def_id::DefIndex::from_u32(0),
};

/// Decoder used to read the offload monomorphization manifest.
pub(crate) struct OffloadManifestDecoder<'a, 'tcx> {
    decoder: MemDecoder<'a>,
    type_shorthands: Lock<FxHashMap<usize, Ty<'tcx>>>,
    #[allow(dead_code)]
    predicate_shorthands: Lock<FxHashMap<usize, ty::PredicateKind<'tcx>>>,
    tcx: TyCtxt<'tcx>,
    /// Map from (crate name, DefPath string) to DefId, used to resolve DefIds
    /// across compilation sessions where StableCrateId differs.
    def_path_map: Lock<Option<FxHashMap<(Symbol, String), DefId>>>,
}

impl<'a, 'tcx> OffloadManifestDecoder<'a, 'tcx> {
    pub(crate) fn new(data: &'a [u8], tcx: TyCtxt<'tcx>) -> Result<Self, ()> {
        let decoder = MemDecoder::new(data, 0)?;
        Ok(OffloadManifestDecoder {
            decoder,
            type_shorthands: Lock::new(FxHashMap::default()),
            predicate_shorthands: Lock::new(FxHashMap::default()),
            tcx,
            def_path_map: Lock::new(None),
        })
    }

    /// (crate name, DefPath) -> DefId map for resolving cross-session DefIds.
    fn get_or_build_def_path_map(&self) -> FxHashMap<(Symbol, String), DefId> {
        let mut guard = self.def_path_map.lock();
        if let Some(map) = guard.as_ref() {
            return map.clone();
        }
        let map = Self::build_def_path_map(self.tcx);
        *guard = Some(map.clone());
        map
    }

    /// Build a (crate name, DefPath) -> DefId map. Owns the format details.
    fn build_def_path_map(tcx: TyCtxt<'tcx>) -> FxHashMap<(Symbol, String), DefId> {
        let mut map: FxHashMap<(Symbol, String), DefId> = FxHashMap::default();

        let local_crate_name = tcx.crate_name(LOCAL_CRATE);
        let krate_items = tcx.hir_crate_items(());
        let local_def_ids = krate_items
            .free_items()
            .map(|id| id.owner_id.to_def_id())
            .chain(krate_items.trait_items().map(|id| id.owner_id.to_def_id()))
            .chain(krate_items.impl_items().map(|id| id.owner_id.to_def_id()))
            .chain(krate_items.foreign_items().map(|id| id.owner_id.to_def_id()));
        for item_id in local_def_ids {
            let def_id = item_id;
            let def_path = tcx.def_path(def_id);
            map.insert((local_crate_name, def_path.to_string_no_crate_verbose()), def_id);
        }

        for &cnum in tcx.crates(()) {
            if cnum == LOCAL_CRATE {
                continue;
            }
            let crate_name = tcx.crate_name(cnum);
            let num_defs = tcx.num_extern_def_ids(cnum);
            for i in 0..num_defs {
                let def_id = DefId { krate: cnum, index: DefIndex::from_usize(i) };
                let def_path = tcx.def_path(def_id);
                map.entry((crate_name, def_path.to_string_no_crate_verbose())).or_insert(def_id);
            }
        }

        map
    }
}

impl<'a, 'tcx> Decoder for OffloadManifestDecoder<'a, 'tcx> {
    fn read_usize(&mut self) -> usize {
        self.decoder.read_usize()
    }
    fn read_u128(&mut self) -> u128 {
        self.decoder.read_u128()
    }
    fn read_u64(&mut self) -> u64 {
        self.decoder.read_u64()
    }
    fn read_u32(&mut self) -> u32 {
        self.decoder.read_u32()
    }
    fn read_u16(&mut self) -> u16 {
        self.decoder.read_u16()
    }
    fn read_u8(&mut self) -> u8 {
        self.decoder.read_u8()
    }
    fn read_isize(&mut self) -> isize {
        self.decoder.read_isize()
    }
    fn read_i128(&mut self) -> i128 {
        self.decoder.read_i128()
    }
    fn read_i64(&mut self) -> i64 {
        self.decoder.read_i64()
    }
    fn read_i32(&mut self) -> i32 {
        self.decoder.read_i32()
    }
    fn read_i16(&mut self) -> i16 {
        self.decoder.read_i16()
    }
    fn read_i8(&mut self) -> i8 {
        self.decoder.read_i8()
    }
    fn read_raw_bytes(&mut self, len: usize) -> &[u8] {
        self.decoder.read_raw_bytes(len)
    }
    fn peek_byte(&self) -> u8 {
        self.decoder.peek_byte()
    }
    fn position(&self) -> usize {
        self.decoder.position()
    }
}

impl<'a, 'tcx> BlobDecoder for OffloadManifestDecoder<'a, 'tcx> {
    fn decode_symbol(&mut self) -> rustc_span::Symbol {
        let s: String = Decodable::decode(self);
        rustc_span::Symbol::intern(&s)
    }

    fn decode_byte_symbol(&mut self) -> ByteSymbol {
        let len = self.read_usize();
        let bytes = self.read_raw_bytes(len);
        ByteSymbol::intern(bytes)
    }

    fn decode_def_index(&mut self) -> rustc_span::def_id::DefIndex {
        let v = self.read_u32();
        rustc_span::def_id::DefIndex::from_u32(v)
    }
}

impl<'a, 'tcx> SpanDecoder for OffloadManifestDecoder<'a, 'tcx> {
    fn decode_span(&mut self) -> Span {
        let lo = self.read_usize();
        let hi = self.read_usize();
        let _ctxt = self.read_u32();
        Span::new(BytePos::from_usize(lo), BytePos::from_usize(hi), SyntaxContext::root(), None)
    }

    fn decode_expn_id(&mut self) -> rustc_span::ExpnId {
        let _ = self.read_u32();
        rustc_span::ExpnId::root()
    }

    fn decode_syntax_context(&mut self) -> SyntaxContext {
        let _ = self.read_u32();
        SyntaxContext::root()
    }

    fn decode_crate_num(&mut self) -> rustc_span::def_id::CrateNum {
        let v = self.read_u32();
        rustc_span::def_id::CrateNum::from_u32(v)
    }

    fn decode_def_id(&mut self) -> rustc_span::def_id::DefId {
        let crate_name: String = Decodable::decode(self);
        let crate_name = Symbol::intern(&crate_name);
        let def_path_str: String = Decodable::decode(self);
        let map = self.get_or_build_def_path_map();
        map.get(&(crate_name, def_path_str)).copied().unwrap_or(UNRESOLVED_DEF_ID)
    }

    fn decode_attr_id(&mut self) -> rustc_ast::AttrId {
        self.tcx.dcx().fatal("AttrIds are not expected in offload manifests");
    }
}

impl<'a, 'tcx> TyDecoder<'tcx> for OffloadManifestDecoder<'a, 'tcx> {
    const CLEAR_CROSS_CRATE: bool = true;

    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn cached_ty_for_shorthand<F>(&mut self, shorthand: usize, or_insert_with: F) -> Ty<'tcx>
    where
        F: FnOnce(&mut Self) -> Ty<'tcx>,
    {
        if let Some(ty) = self.type_shorthands.lock().get(&shorthand) {
            return *ty;
        }
        let ty = or_insert_with(self);
        self.type_shorthands.lock().insert(shorthand, ty);
        ty
    }

    fn with_position<F, R>(&mut self, pos: usize, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let new_decoder = self.decoder.split_at(pos);
        let old_decoder = std::mem::replace(&mut self.decoder, new_decoder);
        let result = f(self);
        self.decoder = old_decoder;
        result
    }

    fn decode_alloc_id(&mut self) -> rustc_middle::mir::interpret::AllocId {
        self.tcx.dcx().fatal("AllocIds are not expected in offload manifests");
    }
}

/// Write a list of offload kernel instances to the manifest file.
pub(crate) fn write_manifest<'tcx>(
    path: &std::path::Path,
    tcx: TyCtxt<'tcx>,
    instances: &[ty::Instance<'tcx>],
) -> std::io::Result<()> {
    let mut encoder = OffloadManifestEncoder::new(path, tcx)?;
    instances.encode(&mut encoder);
    encoder.finish()
}

/// Write out the offload host-metadata manifest for `mono_items`. No-op unless
/// the session was invoked with `-Zoffload=HostMetadata=<path>`.
pub fn write_host_metadata_offload_manifest<'tcx>(tcx: TyCtxt<'tcx>) {
    let Some(path) = tcx.sess.opts.unstable_opts.offload.iter().find_map(|o| {
        if let rustc_session::config::Offload::HostMetadata(p) = o { Some(p) } else { None }
    }) else {
        return;
    };

    let partitions = tcx.collect_and_partition_mono_items(());
    let mono_items: Vec<MonoItem<'_>> = partitions
        .codegen_units
        .iter()
        .flat_map(|cgu| cgu.items().iter())
        .map(|(item, _)| *item)
        .collect();

    let instances: Vec<ty::Instance<'tcx>> = mono_items
        .iter()
        .filter_map(|item| {
            if let MonoItem::Fn(instance) = item {
                if tcx
                    .codegen_fn_attrs(instance.def_id())
                    .flags
                    .contains(CodegenFnAttrFlags::OFFLOAD_KERNEL)
                {
                    Some(*instance)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    if let Err(e) = write_manifest(std::path::Path::new(path), tcx, &instances) {
        tcx.dcx().emit_fatal(crate::diagnostics::OffloadManifestWriteError {
            path: path.clone(),
            err: e.to_string(),
        });
    }
}

/// Read a list of offload kernel instances from the manifest file.
pub(crate) fn read_manifest<'tcx>(
    path: &std::path::Path,
    tcx: TyCtxt<'tcx>,
) -> std::io::Result<Vec<ty::Instance<'tcx>>> {
    let data = fs::read(path)?;
    let mut decoder = OffloadManifestDecoder::new(&data, tcx)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid manifest"))?;

    let payload_len = decoder.decoder.len() - decoder.position();
    if payload_len == 0 {
        return Ok(Vec::new());
    }

    let instances: Vec<ty::Instance<'tcx>> = Decodable::decode(&mut decoder);

    let instances: Vec<_> = instances
        .into_iter()
        .filter(|instance| instance.def_id().krate != rustc_span::def_id::CrateNum::MAX)
        .collect();

    Ok(instances)
}
