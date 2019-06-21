use crate::index::Index;
use crate::schema::*;

use rustc::middle::cstore::{LinkagePreference, NativeLibrary,
                            EncodedMetadata, ForeignModule};
use rustc::hir::def::CtorKind;
use rustc::hir::def_id::{CrateNum, CRATE_DEF_INDEX, DefIndex, DefId, LocalDefId, LOCAL_CRATE};
use rustc::hir::GenericParamKind;
use rustc::hir::map::definitions::DefPathTable;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc::middle::dependency_format::Linkage;
use rustc::middle::exported_symbols::{ExportedSymbol, SymbolExportLevel,
                                      metadata_symbol_name};
use rustc::middle::lang_items;
use rustc::mir::{self, interpret};
use rustc::traits::specialization_graph;
use rustc::ty::{self, Ty, TyCtxt, ReprOptions, SymbolName};
use rustc::ty::codec::{self as ty_codec, TyEncoder};
use rustc::ty::layout::VariantIdx;

use rustc::session::config::{self, CrateType};
use rustc::util::nodemap::FxHashMap;

use rustc_data_structures::stable_hasher::StableHasher;
use rustc_serialize::{Encodable, Encoder, SpecializedEncoder, opaque};

use std::hash::Hash;
use std::path::Path;
use rustc_data_structures::sync::Lrc;
use std::u32;
use syntax::ast;
use syntax::attr;
use syntax::source_map::Spanned;
use syntax::symbol::{kw, sym};
use syntax_pos::{self, FileName, SourceFile, Span};
use log::{debug, trace};

use rustc::hir::{self, PatKind};
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir::intravisit::{Visitor, NestedVisitorMap};
use rustc::hir::intravisit;

pub struct EncodeContext<'tcx> {
    opaque: opaque::Encoder,
    pub tcx: TyCtxt<'tcx>,

    entries_index: Index<'tcx>,

    lazy_state: LazyState,
    type_shorthands: FxHashMap<Ty<'tcx>, usize>,
    predicate_shorthands: FxHashMap<ty::Predicate<'tcx>, usize>,

    interpret_allocs: FxHashMap<interpret::AllocId, usize>,
    interpret_allocs_inverse: Vec<interpret::AllocId>,

    // This is used to speed up Span encoding.
    source_file_cache: Lrc<SourceFile>,
}

macro_rules! encoder_methods {
    ($($name:ident($ty:ty);)*) => {
        $(fn $name(&mut self, value: $ty) -> Result<(), Self::Error> {
            self.opaque.$name(value)
        })*
    }
}

impl<'tcx> Encoder for EncodeContext<'tcx> {
    type Error = <opaque::Encoder as Encoder>::Error;

    fn emit_unit(&mut self) -> Result<(), Self::Error> {
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

impl<'tcx, T> SpecializedEncoder<Lazy<T>> for EncodeContext<'tcx> {
    fn specialized_encode(&mut self, lazy: &Lazy<T>) -> Result<(), Self::Error> {
        self.emit_lazy_distance(lazy.position, Lazy::<T>::min_size())
    }
}

impl<'tcx, T> SpecializedEncoder<LazySeq<T>> for EncodeContext<'tcx> {
    fn specialized_encode(&mut self, seq: &LazySeq<T>) -> Result<(), Self::Error> {
        self.emit_usize(seq.len)?;
        if seq.len == 0 {
            return Ok(());
        }
        self.emit_lazy_distance(seq.position, LazySeq::<T>::min_size(seq.len))
    }
}

impl<'tcx> SpecializedEncoder<CrateNum> for EncodeContext<'tcx> {
    #[inline]
    fn specialized_encode(&mut self, cnum: &CrateNum) -> Result<(), Self::Error> {
        self.emit_u32(cnum.as_u32())
    }
}

impl<'tcx> SpecializedEncoder<DefId> for EncodeContext<'tcx> {
    #[inline]
    fn specialized_encode(&mut self, def_id: &DefId) -> Result<(), Self::Error> {
        let DefId {
            krate,
            index,
        } = *def_id;

        krate.encode(self)?;
        index.encode(self)
    }
}

impl<'tcx> SpecializedEncoder<DefIndex> for EncodeContext<'tcx> {
    #[inline]
    fn specialized_encode(&mut self, def_index: &DefIndex) -> Result<(), Self::Error> {
        self.emit_u32(def_index.as_u32())
    }
}

impl<'tcx> SpecializedEncoder<Span> for EncodeContext<'tcx> {
    fn specialized_encode(&mut self, span: &Span) -> Result<(), Self::Error> {
        if span.is_dummy() {
            return TAG_INVALID_SPAN.encode(self)
        }

        let span = span.data();

        // The Span infrastructure should make sure that this invariant holds:
        debug_assert!(span.lo <= span.hi);

        if !self.source_file_cache.contains(span.lo) {
            let source_map = self.tcx.sess.source_map();
            let source_file_index = source_map.lookup_source_file_idx(span.lo);
            self.source_file_cache = source_map.files()[source_file_index].clone();
        }

        if !self.source_file_cache.contains(span.hi) {
            // Unfortunately, macro expansion still sometimes generates Spans
            // that malformed in this way.
            return TAG_INVALID_SPAN.encode(self)
        }

        TAG_VALID_SPAN.encode(self)?;
        span.lo.encode(self)?;

        // Encode length which is usually less than span.hi and profits more
        // from the variable-length integer encoding that we use.
        let len = span.hi - span.lo;
        len.encode(self)

        // Don't encode the expansion context.
    }
}

impl<'tcx> SpecializedEncoder<LocalDefId> for EncodeContext<'tcx> {
    #[inline]
    fn specialized_encode(&mut self, def_id: &LocalDefId) -> Result<(), Self::Error> {
        self.specialized_encode(&def_id.to_def_id())
    }
}

impl<'tcx> SpecializedEncoder<Ty<'tcx>> for EncodeContext<'tcx> {
    fn specialized_encode(&mut self, ty: &Ty<'tcx>) -> Result<(), Self::Error> {
        ty_codec::encode_with_shorthand(self, ty, |ecx| &mut ecx.type_shorthands)
    }
}

impl<'tcx> SpecializedEncoder<interpret::AllocId> for EncodeContext<'tcx> {
    fn specialized_encode(&mut self, alloc_id: &interpret::AllocId) -> Result<(), Self::Error> {
        use std::collections::hash_map::Entry;
        let index = match self.interpret_allocs.entry(*alloc_id) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let idx = self.interpret_allocs_inverse.len();
                self.interpret_allocs_inverse.push(*alloc_id);
                e.insert(idx);
                idx
            },
        };

        index.encode(self)
    }
}

impl<'tcx> SpecializedEncoder<ty::GenericPredicates<'tcx>> for EncodeContext<'tcx> {
    fn specialized_encode(&mut self,
                          predicates: &ty::GenericPredicates<'tcx>)
                          -> Result<(), Self::Error> {
        ty_codec::encode_predicates(self, predicates, |ecx| &mut ecx.predicate_shorthands)
    }
}

impl<'tcx> SpecializedEncoder<Fingerprint> for EncodeContext<'tcx> {
    fn specialized_encode(&mut self, f: &Fingerprint) -> Result<(), Self::Error> {
        f.encode_opaque(&mut self.opaque)
    }
}

impl<'tcx, T: Encodable> SpecializedEncoder<mir::ClearCrossCrate<T>> for EncodeContext<'tcx> {
    fn specialized_encode(&mut self,
                          _: &mir::ClearCrossCrate<T>)
                          -> Result<(), Self::Error> {
        Ok(())
    }
}

impl<'tcx> TyEncoder for EncodeContext<'tcx> {
    fn position(&self) -> usize {
        self.opaque.position()
    }
}

impl<'tcx> EncodeContext<'tcx> {
    fn emit_node<F: FnOnce(&mut Self, usize) -> R, R>(&mut self, f: F) -> R {
        assert_eq!(self.lazy_state, LazyState::NoNode);
        let pos = self.position();
        self.lazy_state = LazyState::NodeStart(pos);
        let r = f(self, pos);
        self.lazy_state = LazyState::NoNode;
        r
    }

    fn emit_lazy_distance(&mut self,
                          position: usize,
                          min_size: usize)
                          -> Result<(), <Self as Encoder>::Error> {
        let min_end = position + min_size;
        let distance = match self.lazy_state {
            LazyState::NoNode => bug!("emit_lazy_distance: outside of a metadata node"),
            LazyState::NodeStart(start) => {
                assert!(min_end <= start);
                start - min_end
            }
            LazyState::Previous(last_min_end) => {
                assert!(
                    last_min_end <= position,
                    "make sure that the calls to `lazy*` \
                    are in the same order as the metadata fields",
                );
                position - last_min_end
            }
        };
        self.lazy_state = LazyState::Previous(min_end);
        self.emit_usize(distance)
    }

    pub fn lazy<T: Encodable>(&mut self, value: &T) -> Lazy<T> {
        self.emit_node(|ecx, pos| {
            value.encode(ecx).unwrap();

            assert!(pos + Lazy::<T>::min_size() <= ecx.position());
            Lazy::with_position(pos)
        })
    }

    pub fn lazy_seq<I, T>(&mut self, iter: I) -> LazySeq<T>
        where I: IntoIterator<Item = T>,
              T: Encodable
    {
        self.emit_node(|ecx, pos| {
            let len = iter.into_iter().map(|value| value.encode(ecx).unwrap()).count();

            assert!(pos + LazySeq::<T>::min_size(len) <= ecx.position());
            LazySeq::with_position_and_length(pos, len)
        })
    }

    pub fn lazy_seq_ref<'b, I, T>(&mut self, iter: I) -> LazySeq<T>
        where I: IntoIterator<Item = &'b T>,
              T: 'b + Encodable
    {
        self.emit_node(|ecx, pos| {
            let len = iter.into_iter().map(|value| value.encode(ecx).unwrap()).count();

            assert!(pos + LazySeq::<T>::min_size(len) <= ecx.position());
            LazySeq::with_position_and_length(pos, len)
        })
    }

    /// Emit the data for a `DefId` to the metadata. The function to
    /// emit the data is `op`, and it will be given `data` as
    /// arguments. This `record` function will call `op` to generate
    /// the `Entry` (which may point to other encoded information)
    /// and will then record the `Lazy<Entry>` for use in the index.
    // FIXME(eddyb) remove this.
    pub fn record<DATA>(&mut self,
                        id: DefId,
                        op: impl FnOnce(&mut Self, DATA) -> Entry<'tcx>,
                        data: DATA)
    {
        assert!(id.is_local());

        let entry = op(self, data);
        let entry = self.lazy(&entry);
        self.entries_index.record(id, entry);
    }

    fn encode_info_for_items(&mut self) {
        let krate = self.tcx.hir().krate();
        let vis = Spanned { span: syntax_pos::DUMMY_SP, node: hir::VisibilityKind::Public };
        self.record(DefId::local(CRATE_DEF_INDEX),
                     EncodeContext::encode_info_for_mod,
                     (hir::CRATE_HIR_ID, &krate.module, &krate.attrs, &vis));
        krate.visit_all_item_likes(&mut self.as_deep_visitor());
        for macro_def in &krate.exported_macros {
            self.visit_macro_def(macro_def);
        }
    }

    fn encode_def_path_table(&mut self) -> Lazy<DefPathTable> {
        let definitions = self.tcx.hir().definitions();
        self.lazy(definitions.def_path_table())
    }

    fn encode_source_map(&mut self) -> LazySeq<syntax_pos::SourceFile> {
        let source_map = self.tcx.sess.source_map();
        let all_source_files = source_map.files();

        let (working_dir, _cwd_remapped) = self.tcx.sess.working_dir.clone();

        let adapted = all_source_files.iter()
            .filter(|source_file| {
                // No need to re-export imported source_files, as any downstream
                // crate will import them from their original source.
                !source_file.is_imported()
            })
            .map(|source_file| {
                match source_file.name {
                    // This path of this SourceFile has been modified by
                    // path-remapping, so we use it verbatim (and avoid
                    // cloning the whole map in the process).
                    _  if source_file.name_was_remapped => source_file.clone(),

                    // Otherwise expand all paths to absolute paths because
                    // any relative paths are potentially relative to a
                    // wrong directory.
                    FileName::Real(ref name) => {
                        let mut adapted = (**source_file).clone();
                        adapted.name = Path::new(&working_dir).join(name).into();
                        adapted.name_hash = {
                            let mut hasher: StableHasher<u128> = StableHasher::new();
                            adapted.name.hash(&mut hasher);
                            hasher.finish()
                        };
                        Lrc::new(adapted)
                    },

                    // expanded code, not from a file
                    _ => source_file.clone(),
                }
            })
            .collect::<Vec<_>>();

        self.lazy_seq_ref(adapted.iter().map(|rc| &**rc))
    }

    fn encode_crate_root(&mut self) -> Lazy<CrateRoot<'tcx>> {
        let mut i = self.position();

        let crate_deps = self.encode_crate_deps();
        let dylib_dependency_formats = self.encode_dylib_dependency_formats();
        let dep_bytes = self.position() - i;

        // Encode the lib features.
        i = self.position();
        let lib_features = self.encode_lib_features();
        let lib_feature_bytes = self.position() - i;

        // Encode the language items.
        i = self.position();
        let lang_items = self.encode_lang_items();
        let lang_items_missing = self.encode_lang_items_missing();
        let lang_item_bytes = self.position() - i;

        // Encode the native libraries used
        i = self.position();
        let native_libraries = self.encode_native_libraries();
        let native_lib_bytes = self.position() - i;

        let foreign_modules = self.encode_foreign_modules();

        // Encode source_map
        i = self.position();
        let source_map = self.encode_source_map();
        let source_map_bytes = self.position() - i;

        // Encode DefPathTable
        i = self.position();
        let def_path_table = self.encode_def_path_table();
        let def_path_table_bytes = self.position() - i;

        // Encode the def IDs of impls, for coherence checking.
        i = self.position();
        let impls = self.encode_impls();
        let impl_bytes = self.position() - i;

        // Encode exported symbols info.
        i = self.position();
        let exported_symbols = self.tcx.exported_symbols(LOCAL_CRATE);
        let exported_symbols = self.encode_exported_symbols(&exported_symbols);
        let exported_symbols_bytes = self.position() - i;

        let tcx = self.tcx;

        // Encode the items.
        i = self.position();
        self.encode_info_for_items();
        let item_bytes = self.position() - i;

        // Encode the allocation index
        let interpret_alloc_index = {
            let mut interpret_alloc_index = Vec::new();
            let mut n = 0;
            trace!("beginning to encode alloc ids");
            loop {
                let new_n = self.interpret_allocs_inverse.len();
                // if we have found new ids, serialize those, too
                if n == new_n {
                    // otherwise, abort
                    break;
                }
                trace!("encoding {} further alloc ids", new_n - n);
                for idx in n..new_n {
                    let id = self.interpret_allocs_inverse[idx];
                    let pos = self.position() as u32;
                    interpret_alloc_index.push(pos);
                    interpret::specialized_encode_alloc_id(
                        self,
                        tcx,
                        id,
                    ).unwrap();
                }
                n = new_n;
            }
            self.lazy_seq(interpret_alloc_index)
        };

        i = self.position();
        let entries_index = self.entries_index.write_index(&mut self.opaque);
        let entries_index_bytes = self.position() - i;

        let attrs = tcx.hir().krate_attrs();
        let is_proc_macro = tcx.sess.crate_types.borrow().contains(&CrateType::ProcMacro);
        let has_default_lib_allocator = attr::contains_name(&attrs, sym::default_lib_allocator);
        let has_global_allocator = *tcx.sess.has_global_allocator.get();
        let has_panic_handler = *tcx.sess.has_panic_handler.try_get().unwrap_or(&false);

        let root = self.lazy(&CrateRoot {
            name: tcx.crate_name(LOCAL_CRATE),
            extra_filename: tcx.sess.opts.cg.extra_filename.clone(),
            triple: tcx.sess.opts.target_triple.clone(),
            hash: tcx.crate_hash(LOCAL_CRATE),
            disambiguator: tcx.sess.local_crate_disambiguator(),
            panic_strategy: tcx.sess.panic_strategy(),
            edition: tcx.sess.edition(),
            has_global_allocator: has_global_allocator,
            has_panic_handler: has_panic_handler,
            has_default_lib_allocator: has_default_lib_allocator,
            plugin_registrar_fn: tcx.plugin_registrar_fn(LOCAL_CRATE).map(|id| id.index),
            proc_macro_decls_static: if is_proc_macro {
                let id = tcx.proc_macro_decls_static(LOCAL_CRATE).unwrap();
                Some(id.index)
            } else {
                None
            },
            proc_macro_stability: if is_proc_macro {
                tcx.lookup_stability(DefId::local(CRATE_DEF_INDEX)).map(|stab| stab.clone())
            } else {
                None
            },
            compiler_builtins: attr::contains_name(&attrs, sym::compiler_builtins),
            needs_allocator: attr::contains_name(&attrs, sym::needs_allocator),
            needs_panic_runtime: attr::contains_name(&attrs, sym::needs_panic_runtime),
            no_builtins: attr::contains_name(&attrs, sym::no_builtins),
            panic_runtime: attr::contains_name(&attrs, sym::panic_runtime),
            profiler_runtime: attr::contains_name(&attrs, sym::profiler_runtime),
            sanitizer_runtime: attr::contains_name(&attrs, sym::sanitizer_runtime),
            symbol_mangling_version: tcx.sess.opts.debugging_opts.symbol_mangling_version,

            crate_deps,
            dylib_dependency_formats,
            lib_features,
            lang_items,
            lang_items_missing,
            native_libraries,
            foreign_modules,
            source_map,
            def_path_table,
            impls,
            exported_symbols,
            interpret_alloc_index,
            entries_index,
        });

        let total_bytes = self.position();

        if self.tcx.sess.meta_stats() {
            let mut zero_bytes = 0;
            for e in self.opaque.data.iter() {
                if *e == 0 {
                    zero_bytes += 1;
                }
            }

            println!("metadata stats:");
            println!("             dep bytes: {}", dep_bytes);
            println!("     lib feature bytes: {}", lib_feature_bytes);
            println!("       lang item bytes: {}", lang_item_bytes);
            println!("          native bytes: {}", native_lib_bytes);
            println!("         source_map bytes: {}", source_map_bytes);
            println!("            impl bytes: {}", impl_bytes);
            println!("    exp. symbols bytes: {}", exported_symbols_bytes);
            println!("  def-path table bytes: {}", def_path_table_bytes);
            println!("            item bytes: {}", item_bytes);
            println!("   entries index bytes: {}", entries_index_bytes);
            println!("            zero bytes: {}", zero_bytes);
            println!("           total bytes: {}", total_bytes);
        }

        root
    }
}

impl EncodeContext<'tcx> {
    fn encode_variances_of(&mut self, def_id: DefId) -> LazySeq<ty::Variance> {
        debug!("EncodeContext::encode_variances_of({:?})", def_id);
        let tcx = self.tcx;
        self.lazy_seq_ref(&tcx.variances_of(def_id)[..])
    }

    fn encode_item_type(&mut self, def_id: DefId) -> Lazy<Ty<'tcx>> {
        let tcx = self.tcx;
        let ty = tcx.type_of(def_id);
        debug!("EncodeContext::encode_item_type({:?}) => {:?}", def_id, ty);
        self.lazy(&ty)
    }

    fn encode_enum_variant_info(
        &mut self,
        (enum_did, index): (DefId, VariantIdx),
    ) -> Entry<'tcx> {
        let tcx = self.tcx;
        let def = tcx.adt_def(enum_did);
        let variant = &def.variants[index];
        let def_id = variant.def_id;
        debug!("EncodeContext::encode_enum_variant_info({:?})", def_id);

        let data = VariantData {
            ctor_kind: variant.ctor_kind,
            discr: variant.discr,
            // FIXME(eddyb) deduplicate these with `encode_enum_variant_ctor`.
            ctor: variant.ctor_def_id.map(|did| did.index),
            ctor_sig: if variant.ctor_kind == CtorKind::Fn {
                variant.ctor_def_id.map(|ctor_def_id| self.lazy(&tcx.fn_sig(ctor_def_id)))
            } else {
                None
            },
        };

        let enum_id = tcx.hir().as_local_hir_id(enum_did).unwrap();
        let enum_vis = &tcx.hir().expect_item(enum_id).vis;

        Entry {
            kind: EntryKind::Variant(self.lazy(&data)),
            visibility: self.lazy(&ty::Visibility::from_hir(enum_vis, enum_id, tcx)),
            span: self.lazy(&tcx.def_span(def_id)),
            attributes: self.encode_attributes(&tcx.get_attrs(def_id)),
            children: self.lazy_seq(variant.fields.iter().map(|f| {
                assert!(f.did.is_local());
                f.did.index
            })),
            stability: self.encode_stability(def_id),
            deprecation: self.encode_deprecation(def_id),

            ty: Some(self.encode_item_type(def_id)),
            inherent_impls: LazySeq::empty(),
            variances: if variant.ctor_kind == CtorKind::Fn {
                self.encode_variances_of(def_id)
            } else {
                LazySeq::empty()
            },
            generics: Some(self.encode_generics(def_id)),
            predicates: Some(self.encode_predicates(def_id)),
            predicates_defined_on: None,

            mir: self.encode_optimized_mir(def_id),
        }
    }

    fn encode_enum_variant_ctor(
        &mut self,
        (enum_did, index): (DefId, VariantIdx),
    ) -> Entry<'tcx> {
        let tcx = self.tcx;
        let def = tcx.adt_def(enum_did);
        let variant = &def.variants[index];
        let def_id = variant.ctor_def_id.unwrap();
        debug!("EncodeContext::encode_enum_variant_ctor({:?})", def_id);

        let data = VariantData {
            ctor_kind: variant.ctor_kind,
            discr: variant.discr,
            ctor: Some(def_id.index),
            ctor_sig: if variant.ctor_kind == CtorKind::Fn {
                Some(self.lazy(&tcx.fn_sig(def_id)))
            } else {
                None
            }
        };

        // Variant constructors have the same visibility as the parent enums, unless marked as
        // non-exhaustive, in which case they are lowered to `pub(crate)`.
        let enum_id = tcx.hir().as_local_hir_id(enum_did).unwrap();
        let enum_vis = &tcx.hir().expect_item(enum_id).vis;
        let mut ctor_vis = ty::Visibility::from_hir(enum_vis, enum_id, tcx);
        if variant.is_field_list_non_exhaustive() && ctor_vis == ty::Visibility::Public {
            ctor_vis = ty::Visibility::Restricted(DefId::local(CRATE_DEF_INDEX));
        }

        Entry {
            kind: EntryKind::Variant(self.lazy(&data)),
            visibility: self.lazy(&ctor_vis),
            span: self.lazy(&tcx.def_span(def_id)),
            attributes: LazySeq::empty(),
            children: LazySeq::empty(),
            stability: self.encode_stability(def_id),
            deprecation: self.encode_deprecation(def_id),

            ty: Some(self.encode_item_type(def_id)),
            inherent_impls: LazySeq::empty(),
            variances: if variant.ctor_kind == CtorKind::Fn {
                self.encode_variances_of(def_id)
            } else {
                LazySeq::empty()
            },
            generics: Some(self.encode_generics(def_id)),
            predicates: Some(self.encode_predicates(def_id)),
            predicates_defined_on: None,

            mir: self.encode_optimized_mir(def_id),
        }
    }

    fn encode_info_for_mod(
        &mut self,
        (id, md, attrs, vis): (hir::HirId, &hir::Mod, &[ast::Attribute], &hir::Visibility),
    ) -> Entry<'tcx> {
        let tcx = self.tcx;
        let def_id = tcx.hir().local_def_id_from_hir_id(id);
        debug!("EncodeContext::encode_info_for_mod({:?})", def_id);

        let data = ModData {
            reexports: match tcx.module_exports(def_id) {
                Some(exports) => self.lazy_seq_ref(exports),
                _ => LazySeq::empty(),
            },
        };

        Entry {
            kind: EntryKind::Mod(self.lazy(&data)),
            visibility: self.lazy(&ty::Visibility::from_hir(vis, id, tcx)),
            span: self.lazy(&tcx.def_span(def_id)),
            attributes: self.encode_attributes(attrs),
            children: self.lazy_seq(md.item_ids.iter().map(|item_id| {
                tcx.hir().local_def_id_from_hir_id(item_id.id).index
            })),
            stability: self.encode_stability(def_id),
            deprecation: self.encode_deprecation(def_id),

            ty: None,
            inherent_impls: LazySeq::empty(),
            variances: LazySeq::empty(),
            generics: None,
            predicates: None,
            predicates_defined_on: None,

            mir: None
        }
    }

    fn encode_field(
        &mut self,
        (adt_def_id, variant_index, field_index): (DefId, VariantIdx, usize),
    ) -> Entry<'tcx> {
        let tcx = self.tcx;
        let variant = &tcx.adt_def(adt_def_id).variants[variant_index];
        let field = &variant.fields[field_index];

        let def_id = field.did;
        debug!("EncodeContext::encode_field({:?})", def_id);

        let variant_id = tcx.hir().as_local_hir_id(variant.def_id).unwrap();
        let variant_data = tcx.hir().expect_variant_data(variant_id);

        Entry {
            kind: EntryKind::Field,
            visibility: self.lazy(&field.vis),
            span: self.lazy(&tcx.def_span(def_id)),
            attributes: self.encode_attributes(&variant_data.fields()[field_index].attrs),
            children: LazySeq::empty(),
            stability: self.encode_stability(def_id),
            deprecation: self.encode_deprecation(def_id),

            ty: Some(self.encode_item_type(def_id)),
            inherent_impls: LazySeq::empty(),
            variances: LazySeq::empty(),
            generics: Some(self.encode_generics(def_id)),
            predicates: Some(self.encode_predicates(def_id)),
            predicates_defined_on: None,

            mir: None,
        }
    }

    fn encode_struct_ctor(&mut self, (adt_def_id, def_id): (DefId, DefId)) -> Entry<'tcx> {
        debug!("EncodeContext::encode_struct_ctor({:?})", def_id);
        let tcx = self.tcx;
        let adt_def = tcx.adt_def(adt_def_id);
        let variant = adt_def.non_enum_variant();

        let data = VariantData {
            ctor_kind: variant.ctor_kind,
            discr: variant.discr,
            ctor: Some(def_id.index),
            ctor_sig: if variant.ctor_kind == CtorKind::Fn {
                Some(self.lazy(&tcx.fn_sig(def_id)))
            } else {
                None
            }
        };

        let struct_id = tcx.hir().as_local_hir_id(adt_def_id).unwrap();
        let struct_vis = &tcx.hir().expect_item(struct_id).vis;
        let mut ctor_vis = ty::Visibility::from_hir(struct_vis, struct_id, tcx);
        for field in &variant.fields {
            if ctor_vis.is_at_least(field.vis, tcx) {
                ctor_vis = field.vis;
            }
        }

        // If the structure is marked as non_exhaustive then lower the visibility
        // to within the crate.
        if adt_def.non_enum_variant().is_field_list_non_exhaustive() &&
            ctor_vis == ty::Visibility::Public
        {
            ctor_vis = ty::Visibility::Restricted(DefId::local(CRATE_DEF_INDEX));
        }

        let repr_options = get_repr_options(tcx, adt_def_id);

        Entry {
            kind: EntryKind::Struct(self.lazy(&data), repr_options),
            visibility: self.lazy(&ctor_vis),
            span: self.lazy(&tcx.def_span(def_id)),
            attributes: LazySeq::empty(),
            children: LazySeq::empty(),
            stability: self.encode_stability(def_id),
            deprecation: self.encode_deprecation(def_id),

            ty: Some(self.encode_item_type(def_id)),
            inherent_impls: LazySeq::empty(),
            variances: if variant.ctor_kind == CtorKind::Fn {
                self.encode_variances_of(def_id)
            } else {
                LazySeq::empty()
            },
            generics: Some(self.encode_generics(def_id)),
            predicates: Some(self.encode_predicates(def_id)),
            predicates_defined_on: None,

            mir: self.encode_optimized_mir(def_id),
        }
    }

    fn encode_generics(&mut self, def_id: DefId) -> Lazy<ty::Generics> {
        debug!("EncodeContext::encode_generics({:?})", def_id);
        let tcx = self.tcx;
        self.lazy(tcx.generics_of(def_id))
    }

    fn encode_predicates(&mut self, def_id: DefId) -> Lazy<ty::GenericPredicates<'tcx>> {
        debug!("EncodeContext::encode_predicates({:?})", def_id);
        let tcx = self.tcx;
        self.lazy(&tcx.predicates_of(def_id))
    }

    fn encode_predicates_defined_on(&mut self, def_id: DefId) -> Lazy<ty::GenericPredicates<'tcx>> {
        debug!("EncodeContext::encode_predicates_defined_on({:?})", def_id);
        let tcx = self.tcx;
        self.lazy(&tcx.predicates_defined_on(def_id))
    }

    fn encode_info_for_trait_item(&mut self, def_id: DefId) -> Entry<'tcx> {
        debug!("EncodeContext::encode_info_for_trait_item({:?})", def_id);
        let tcx = self.tcx;

        let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();
        let ast_item = tcx.hir().expect_trait_item(hir_id);
        let trait_item = tcx.associated_item(def_id);

        let container = match trait_item.defaultness {
            hir::Defaultness::Default { has_value: true } =>
                AssocContainer::TraitWithDefault,
            hir::Defaultness::Default { has_value: false } =>
                AssocContainer::TraitRequired,
            hir::Defaultness::Final =>
                span_bug!(ast_item.span, "traits cannot have final items"),
        };

        let kind = match trait_item.kind {
            ty::AssocKind::Const => {
                let const_qualif =
                    if let hir::TraitItemKind::Const(_, Some(body)) = ast_item.node {
                        self.const_qualif(0, body)
                    } else {
                        ConstQualif { mir: 0, ast_promotable: false }
                    };

                let rendered =
                    hir::print::to_string(self.tcx.hir(), |s| s.print_trait_item(ast_item));
                let rendered_const = self.lazy(&RenderedConst(rendered));

                EntryKind::AssocConst(container, const_qualif, rendered_const)
            }
            ty::AssocKind::Method => {
                let fn_data = if let hir::TraitItemKind::Method(_, ref m) = ast_item.node {
                    let arg_names = match *m {
                        hir::TraitMethod::Required(ref names) => {
                            self.encode_fn_arg_names(names)
                        }
                        hir::TraitMethod::Provided(body) => {
                            self.encode_fn_arg_names_for_body(body)
                        }
                    };
                    FnData {
                        constness: hir::Constness::NotConst,
                        arg_names,
                        sig: self.lazy(&tcx.fn_sig(def_id)),
                    }
                } else {
                    bug!()
                };
                EntryKind::Method(self.lazy(&MethodData {
                    fn_data,
                    container,
                    has_self: trait_item.method_has_self_argument,
                }))
            }
            ty::AssocKind::Type => EntryKind::AssocType(container),
            ty::AssocKind::Existential =>
                span_bug!(ast_item.span, "existential type in trait"),
        };

        Entry {
            kind,
            visibility: self.lazy(&trait_item.vis),
            span: self.lazy(&ast_item.span),
            attributes: self.encode_attributes(&ast_item.attrs),
            children: LazySeq::empty(),
            stability: self.encode_stability(def_id),
            deprecation: self.encode_deprecation(def_id),

            ty: match trait_item.kind {
                ty::AssocKind::Const |
                ty::AssocKind::Method => {
                    Some(self.encode_item_type(def_id))
                }
                ty::AssocKind::Type => {
                    if trait_item.defaultness.has_value() {
                        Some(self.encode_item_type(def_id))
                    } else {
                        None
                    }
                }
                ty::AssocKind::Existential => unreachable!(),
            },
            inherent_impls: LazySeq::empty(),
            variances: if trait_item.kind == ty::AssocKind::Method {
                self.encode_variances_of(def_id)
            } else {
                LazySeq::empty()
            },
            generics: Some(self.encode_generics(def_id)),
            predicates: Some(self.encode_predicates(def_id)),
            predicates_defined_on: None,

            mir: self.encode_optimized_mir(def_id),
        }
    }

    fn metadata_output_only(&self) -> bool {
        // MIR optimisation can be skipped when we're just interested in the metadata.
        !self.tcx.sess.opts.output_types.should_codegen()
    }

    fn const_qualif(&self, mir: u8, body_id: hir::BodyId) -> ConstQualif {
        let body_owner_def_id = self.tcx.hir().body_owner_def_id(body_id);
        let ast_promotable = self.tcx.const_is_rvalue_promotable_to_static(body_owner_def_id);

        ConstQualif { mir, ast_promotable }
    }

    fn encode_info_for_impl_item(&mut self, def_id: DefId) -> Entry<'tcx> {
        debug!("EncodeContext::encode_info_for_impl_item({:?})", def_id);
        let tcx = self.tcx;

        let hir_id = self.tcx.hir().as_local_hir_id(def_id).unwrap();
        let ast_item = self.tcx.hir().expect_impl_item(hir_id);
        let impl_item = self.tcx.associated_item(def_id);

        let container = match impl_item.defaultness {
            hir::Defaultness::Default { has_value: true } => AssocContainer::ImplDefault,
            hir::Defaultness::Final => AssocContainer::ImplFinal,
            hir::Defaultness::Default { has_value: false } =>
                span_bug!(ast_item.span, "impl items always have values (currently)"),
        };

        let kind = match impl_item.kind {
            ty::AssocKind::Const => {
                if let hir::ImplItemKind::Const(_, body_id) = ast_item.node {
                    let mir = self.tcx.at(ast_item.span).mir_const_qualif(def_id).0;

                    EntryKind::AssocConst(container,
                        self.const_qualif(mir, body_id),
                        self.encode_rendered_const_for_body(body_id))
                } else {
                    bug!()
                }
            }
            ty::AssocKind::Method => {
                let fn_data = if let hir::ImplItemKind::Method(ref sig, body) = ast_item.node {
                    FnData {
                        constness: sig.header.constness,
                        arg_names: self.encode_fn_arg_names_for_body(body),
                        sig: self.lazy(&tcx.fn_sig(def_id)),
                    }
                } else {
                    bug!()
                };
                EntryKind::Method(self.lazy(&MethodData {
                    fn_data,
                    container,
                    has_self: impl_item.method_has_self_argument,
                }))
            }
            ty::AssocKind::Existential => EntryKind::AssocExistential(container),
            ty::AssocKind::Type => EntryKind::AssocType(container)
        };

        let mir =
            match ast_item.node {
                hir::ImplItemKind::Const(..) => true,
                hir::ImplItemKind::Method(ref sig, _) => {
                    let generics = self.tcx.generics_of(def_id);
                    let needs_inline = (generics.requires_monomorphization(self.tcx) ||
                                        tcx.codegen_fn_attrs(def_id).requests_inline()) &&
                                        !self.metadata_output_only();
                    let is_const_fn = sig.header.constness == hir::Constness::Const;
                    let always_encode_mir = self.tcx.sess.opts.debugging_opts.always_encode_mir;
                    needs_inline || is_const_fn || always_encode_mir
                },
                hir::ImplItemKind::Existential(..) |
                hir::ImplItemKind::Type(..) => false,
            };

        Entry {
            kind,
            visibility: self.lazy(&impl_item.vis),
            span: self.lazy(&ast_item.span),
            attributes: self.encode_attributes(&ast_item.attrs),
            children: LazySeq::empty(),
            stability: self.encode_stability(def_id),
            deprecation: self.encode_deprecation(def_id),

            ty: Some(self.encode_item_type(def_id)),
            inherent_impls: LazySeq::empty(),
            variances: if impl_item.kind == ty::AssocKind::Method {
                self.encode_variances_of(def_id)
            } else {
                LazySeq::empty()
            },
            generics: Some(self.encode_generics(def_id)),
            predicates: Some(self.encode_predicates(def_id)),
            predicates_defined_on: None,

            mir: if mir { self.encode_optimized_mir(def_id) } else { None },
        }
    }

    fn encode_fn_arg_names_for_body(&mut self, body_id: hir::BodyId)
                                    -> LazySeq<ast::Name> {
        self.tcx.dep_graph.with_ignore(|| {
            let body = self.tcx.hir().body(body_id);
            self.lazy_seq(body.arguments.iter().map(|arg| {
                match arg.pat.node {
                    PatKind::Binding(_, _, ident, _) => ident.name,
                    _ => kw::Invalid,
                }
            }))
        })
    }

    fn encode_fn_arg_names(&mut self, param_names: &[ast::Ident]) -> LazySeq<ast::Name> {
        self.lazy_seq(param_names.iter().map(|ident| ident.name))
    }

    fn encode_optimized_mir(&mut self, def_id: DefId) -> Option<Lazy<mir::Body<'tcx>>> {
        debug!("EntryBuilder::encode_mir({:?})", def_id);
        if self.tcx.mir_keys(LOCAL_CRATE).contains(&def_id) {
            let mir = self.tcx.optimized_mir(def_id);
            Some(self.lazy(&mir))
        } else {
            None
        }
    }

    // Encodes the inherent implementations of a structure, enumeration, or trait.
    fn encode_inherent_implementations(&mut self, def_id: DefId) -> LazySeq<DefIndex> {
        debug!("EncodeContext::encode_inherent_implementations({:?})", def_id);
        let implementations = self.tcx.inherent_impls(def_id);
        if implementations.is_empty() {
            LazySeq::empty()
        } else {
            self.lazy_seq(implementations.iter().map(|&def_id| {
                assert!(def_id.is_local());
                def_id.index
            }))
        }
    }

    fn encode_stability(&mut self, def_id: DefId) -> Option<Lazy<attr::Stability>> {
        debug!("EncodeContext::encode_stability({:?})", def_id);
        self.tcx.lookup_stability(def_id).map(|stab| self.lazy(stab))
    }

    fn encode_deprecation(&mut self, def_id: DefId) -> Option<Lazy<attr::Deprecation>> {
        debug!("EncodeContext::encode_deprecation({:?})", def_id);
        self.tcx.lookup_deprecation(def_id).map(|depr| self.lazy(&depr))
    }

    fn encode_rendered_const_for_body(&mut self, body_id: hir::BodyId) -> Lazy<RenderedConst> {
        let body = self.tcx.hir().body(body_id);
        let rendered = hir::print::to_string(self.tcx.hir(), |s| s.print_expr(&body.value));
        let rendered_const = &RenderedConst(rendered);
        self.lazy(rendered_const)
    }

    fn encode_info_for_item(&mut self, (def_id, item): (DefId, &'tcx hir::Item)) -> Entry<'tcx> {
        let tcx = self.tcx;

        debug!("EncodeContext::encode_info_for_item({:?})", def_id);

        let kind = match item.node {
            hir::ItemKind::Static(_, hir::MutMutable, _) => EntryKind::MutStatic,
            hir::ItemKind::Static(_, hir::MutImmutable, _) => EntryKind::ImmStatic,
            hir::ItemKind::Const(_, body_id) => {
                let mir = tcx.at(item.span).mir_const_qualif(def_id).0;
                EntryKind::Const(
                    self.const_qualif(mir, body_id),
                    self.encode_rendered_const_for_body(body_id)
                )
            }
            hir::ItemKind::Fn(_, header, .., body) => {
                let data = FnData {
                    constness: header.constness,
                    arg_names: self.encode_fn_arg_names_for_body(body),
                    sig: self.lazy(&tcx.fn_sig(def_id)),
                };

                EntryKind::Fn(self.lazy(&data))
            }
            hir::ItemKind::Mod(ref m) => {
                return self.encode_info_for_mod((item.hir_id, m, &item.attrs, &item.vis));
            }
            hir::ItemKind::ForeignMod(_) => EntryKind::ForeignMod,
            hir::ItemKind::GlobalAsm(..) => EntryKind::GlobalAsm,
            hir::ItemKind::Ty(..) => EntryKind::Type,
            hir::ItemKind::Existential(..) => EntryKind::Existential,
            hir::ItemKind::Enum(..) => EntryKind::Enum(get_repr_options(tcx, def_id)),
            hir::ItemKind::Struct(ref struct_def, _) => {
                let variant = tcx.adt_def(def_id).non_enum_variant();

                // Encode def_ids for each field and method
                // for methods, write all the stuff get_trait_method
                // needs to know
                let ctor = struct_def.ctor_hir_id()
                    .map(|ctor_hir_id| tcx.hir().local_def_id_from_hir_id(ctor_hir_id).index);

                let repr_options = get_repr_options(tcx, def_id);

                EntryKind::Struct(self.lazy(&VariantData {
                    ctor_kind: variant.ctor_kind,
                    discr: variant.discr,
                    ctor,
                    ctor_sig: None,
                }), repr_options)
            }
            hir::ItemKind::Union(..) => {
                let variant = tcx.adt_def(def_id).non_enum_variant();
                let repr_options = get_repr_options(tcx, def_id);

                EntryKind::Union(self.lazy(&VariantData {
                    ctor_kind: variant.ctor_kind,
                    discr: variant.discr,
                    ctor: None,
                    ctor_sig: None,
                }), repr_options)
            }
            hir::ItemKind::Impl(_, polarity, defaultness, ..) => {
                let trait_ref = tcx.impl_trait_ref(def_id);
                let parent = if let Some(trait_ref) = trait_ref {
                    let trait_def = tcx.trait_def(trait_ref.def_id);
                    trait_def.ancestors(tcx, def_id).nth(1).and_then(|node| {
                        match node {
                            specialization_graph::Node::Impl(parent) => Some(parent),
                            _ => None,
                        }
                    })
                } else {
                    None
                };

                // if this is an impl of `CoerceUnsized`, create its
                // "unsized info", else just store None
                let coerce_unsized_info =
                    trait_ref.and_then(|t| {
                        if Some(t.def_id) == tcx.lang_items().coerce_unsized_trait() {
                            Some(tcx.at(item.span).coerce_unsized_info(def_id))
                        } else {
                            None
                        }
                    });

                let data = ImplData {
                    polarity,
                    defaultness,
                    parent_impl: parent,
                    coerce_unsized_info,
                    trait_ref: trait_ref.map(|trait_ref| self.lazy(&trait_ref)),
                };

                EntryKind::Impl(self.lazy(&data))
            }
            hir::ItemKind::Trait(..) => {
                let trait_def = tcx.trait_def(def_id);
                let data = TraitData {
                    unsafety: trait_def.unsafety,
                    paren_sugar: trait_def.paren_sugar,
                    has_auto_impl: tcx.trait_is_auto(def_id),
                    is_marker: trait_def.is_marker,
                    super_predicates: self.lazy(&tcx.super_predicates_of(def_id)),
                };

                EntryKind::Trait(self.lazy(&data))
            }
            hir::ItemKind::TraitAlias(..) => {
                let data = TraitAliasData {
                    super_predicates: self.lazy(&tcx.super_predicates_of(def_id)),
                };

                EntryKind::TraitAlias(self.lazy(&data))
            }
            hir::ItemKind::ExternCrate(_) |
            hir::ItemKind::Use(..) => bug!("cannot encode info for item {:?}", item),
        };

        Entry {
            kind,
            visibility: self.lazy(&ty::Visibility::from_hir(&item.vis, item.hir_id, tcx)),
            span: self.lazy(&item.span),
            attributes: self.encode_attributes(&item.attrs),
            children: match item.node {
                hir::ItemKind::ForeignMod(ref fm) => {
                    self.lazy_seq(fm.items
                        .iter()
                        .map(|foreign_item| tcx.hir().local_def_id_from_hir_id(
                            foreign_item.hir_id).index))
                }
                hir::ItemKind::Enum(..) => {
                    let def = self.tcx.adt_def(def_id);
                    self.lazy_seq(def.variants.iter().map(|v| {
                        assert!(v.def_id.is_local());
                        v.def_id.index
                    }))
                }
                hir::ItemKind::Struct(..) |
                hir::ItemKind::Union(..) => {
                    let def = self.tcx.adt_def(def_id);
                    self.lazy_seq(def.non_enum_variant().fields.iter().map(|f| {
                        assert!(f.did.is_local());
                        f.did.index
                    }))
                }
                hir::ItemKind::Impl(..) |
                hir::ItemKind::Trait(..) => {
                    self.lazy_seq(tcx.associated_item_def_ids(def_id).iter().map(|&def_id| {
                        assert!(def_id.is_local());
                        def_id.index
                    }))
                }
                _ => LazySeq::empty(),
            },
            stability: self.encode_stability(def_id),
            deprecation: self.encode_deprecation(def_id),

            ty: match item.node {
                hir::ItemKind::Static(..) |
                hir::ItemKind::Const(..) |
                hir::ItemKind::Fn(..) |
                hir::ItemKind::Ty(..) |
                hir::ItemKind::Existential(..) |
                hir::ItemKind::Enum(..) |
                hir::ItemKind::Struct(..) |
                hir::ItemKind::Union(..) |
                hir::ItemKind::Impl(..) => Some(self.encode_item_type(def_id)),
                _ => None,
            },
            inherent_impls: self.encode_inherent_implementations(def_id),
            variances: match item.node {
                hir::ItemKind::Enum(..) |
                hir::ItemKind::Struct(..) |
                hir::ItemKind::Union(..) |
                hir::ItemKind::Fn(..) => self.encode_variances_of(def_id),
                _ => LazySeq::empty(),
            },
            generics: match item.node {
                hir::ItemKind::Static(..) |
                hir::ItemKind::Const(..) |
                hir::ItemKind::Fn(..) |
                hir::ItemKind::Ty(..) |
                hir::ItemKind::Enum(..) |
                hir::ItemKind::Struct(..) |
                hir::ItemKind::Union(..) |
                hir::ItemKind::Impl(..) |
                hir::ItemKind::Existential(..) |
                hir::ItemKind::Trait(..) => Some(self.encode_generics(def_id)),
                hir::ItemKind::TraitAlias(..) => Some(self.encode_generics(def_id)),
                _ => None,
            },
            predicates: match item.node {
                hir::ItemKind::Static(..) |
                hir::ItemKind::Const(..) |
                hir::ItemKind::Fn(..) |
                hir::ItemKind::Ty(..) |
                hir::ItemKind::Enum(..) |
                hir::ItemKind::Struct(..) |
                hir::ItemKind::Union(..) |
                hir::ItemKind::Impl(..) |
                hir::ItemKind::Existential(..) |
                hir::ItemKind::Trait(..) |
                hir::ItemKind::TraitAlias(..) => Some(self.encode_predicates(def_id)),
                _ => None,
            },

            // The only time that `predicates_defined_on` is used (on
            // an external item) is for traits, during chalk lowering,
            // so only encode it in that case as an efficiency
            // hack. (No reason not to expand it in the future if
            // necessary.)
            predicates_defined_on: match item.node {
                hir::ItemKind::Trait(..) |
                hir::ItemKind::TraitAlias(..) => Some(self.encode_predicates_defined_on(def_id)),
                _ => None, // not *wrong* for other kinds of items, but not needed
            },

            mir: match item.node {
                hir::ItemKind::Static(..) => {
                    self.encode_optimized_mir(def_id)
                }
                hir::ItemKind::Const(..) => self.encode_optimized_mir(def_id),
                hir::ItemKind::Fn(_, header, ..) => {
                    let generics = tcx.generics_of(def_id);
                    let needs_inline =
                        (generics.requires_monomorphization(tcx) ||
                         tcx.codegen_fn_attrs(def_id).requests_inline()) &&
                            !self.metadata_output_only();
                    let always_encode_mir = self.tcx.sess.opts.debugging_opts.always_encode_mir;
                    if needs_inline
                        || header.constness == hir::Constness::Const
                        || always_encode_mir
                    {
                        self.encode_optimized_mir(def_id)
                    } else {
                        None
                    }
                }
                _ => None,
            },
        }
    }

    /// Serialize the text of exported macros
    fn encode_info_for_macro_def(&mut self, macro_def: &hir::MacroDef) -> Entry<'tcx> {
        use syntax::print::pprust;
        let def_id = self.tcx.hir().local_def_id_from_hir_id(macro_def.hir_id);
        Entry {
            kind: EntryKind::MacroDef(self.lazy(&MacroDef {
                body: pprust::tts_to_string(&macro_def.body.trees().collect::<Vec<_>>()),
                legacy: macro_def.legacy,
            })),
            visibility: self.lazy(&ty::Visibility::Public),
            span: self.lazy(&macro_def.span),
            attributes: self.encode_attributes(&macro_def.attrs),
            stability: self.encode_stability(def_id),
            deprecation: self.encode_deprecation(def_id),

            children: LazySeq::empty(),
            ty: None,
            inherent_impls: LazySeq::empty(),
            variances: LazySeq::empty(),
            generics: None,
            predicates: None,
            predicates_defined_on: None,
            mir: None,
        }
    }

    fn encode_info_for_generic_param(
        &mut self,
        def_id: DefId,
        entry_kind: EntryKind<'tcx>,
        encode_type: bool,
    ) -> Entry<'tcx> {
        let tcx = self.tcx;
        Entry {
            kind: entry_kind,
            visibility: self.lazy(&ty::Visibility::Public),
            span: self.lazy(&tcx.def_span(def_id)),
            attributes: LazySeq::empty(),
            children: LazySeq::empty(),
            stability: None,
            deprecation: None,
            ty: if encode_type { Some(self.encode_item_type(def_id)) } else { None },
            inherent_impls: LazySeq::empty(),
            variances: LazySeq::empty(),
            generics: None,
            predicates: None,
            predicates_defined_on: None,

            mir: None,
        }
    }

    fn encode_info_for_ty_param(
        &mut self,
        (def_id, encode_type): (DefId, bool),
    ) -> Entry<'tcx> {
        debug!("EncodeContext::encode_info_for_ty_param({:?})", def_id);
        self.encode_info_for_generic_param(def_id, EntryKind::TypeParam, encode_type)
    }

    fn encode_info_for_const_param(
        &mut self,
        def_id: DefId,
    ) -> Entry<'tcx> {
        debug!("EncodeContext::encode_info_for_const_param({:?})", def_id);
        self.encode_info_for_generic_param(def_id, EntryKind::ConstParam, true)
    }

    fn encode_info_for_closure(&mut self, def_id: DefId) -> Entry<'tcx> {
        debug!("EncodeContext::encode_info_for_closure({:?})", def_id);
        let tcx = self.tcx;

        let tables = self.tcx.typeck_tables_of(def_id);
        let hir_id = self.tcx.hir().as_local_hir_id(def_id).unwrap();
        let kind = match tables.node_type(hir_id).sty {
            ty::Generator(def_id, ..) => {
                let layout = self.tcx.generator_layout(def_id);
                let data = GeneratorData {
                    layout: layout.clone(),
                };
                EntryKind::Generator(self.lazy(&data))
            }

            ty::Closure(def_id, substs) => {
                let sig = substs.closure_sig(def_id, self.tcx);
                let data = ClosureData { sig: self.lazy(&sig) };
                EntryKind::Closure(self.lazy(&data))
            }

            _ => bug!("closure that is neither generator nor closure")
        };

        Entry {
            kind,
            visibility: self.lazy(&ty::Visibility::Public),
            span: self.lazy(&tcx.def_span(def_id)),
            attributes: self.encode_attributes(&tcx.get_attrs(def_id)),
            children: LazySeq::empty(),
            stability: None,
            deprecation: None,

            ty: Some(self.encode_item_type(def_id)),
            inherent_impls: LazySeq::empty(),
            variances: LazySeq::empty(),
            generics: Some(self.encode_generics(def_id)),
            predicates: None,
            predicates_defined_on: None,

            mir: self.encode_optimized_mir(def_id),
        }
    }

    fn encode_info_for_anon_const(&mut self, def_id: DefId) -> Entry<'tcx> {
        debug!("EncodeContext::encode_info_for_anon_const({:?})", def_id);
        let tcx = self.tcx;
        let id = tcx.hir().as_local_hir_id(def_id).unwrap();
        let body_id = tcx.hir().body_owned_by(id);
        let const_data = self.encode_rendered_const_for_body(body_id);
        let mir = tcx.mir_const_qualif(def_id).0;

        Entry {
            kind: EntryKind::Const(self.const_qualif(mir, body_id), const_data),
            visibility: self.lazy(&ty::Visibility::Public),
            span: self.lazy(&tcx.def_span(def_id)),
            attributes: LazySeq::empty(),
            children: LazySeq::empty(),
            stability: None,
            deprecation: None,

            ty: Some(self.encode_item_type(def_id)),
            inherent_impls: LazySeq::empty(),
            variances: LazySeq::empty(),
            generics: Some(self.encode_generics(def_id)),
            predicates: Some(self.encode_predicates(def_id)),
            predicates_defined_on: None,

            mir: self.encode_optimized_mir(def_id),
        }
    }

    fn encode_attributes(&mut self, attrs: &[ast::Attribute]) -> LazySeq<ast::Attribute> {
        self.lazy_seq_ref(attrs)
    }

    fn encode_native_libraries(&mut self) -> LazySeq<NativeLibrary> {
        let used_libraries = self.tcx.native_libraries(LOCAL_CRATE);
        self.lazy_seq(used_libraries.iter().cloned())
    }

    fn encode_foreign_modules(&mut self) -> LazySeq<ForeignModule> {
        let foreign_modules = self.tcx.foreign_modules(LOCAL_CRATE);
        self.lazy_seq(foreign_modules.iter().cloned())
    }

    fn encode_crate_deps(&mut self) -> LazySeq<CrateDep> {
        let crates = self.tcx.crates();

        let mut deps = crates
            .iter()
            .map(|&cnum| {
                let dep = CrateDep {
                    name: self.tcx.original_crate_name(cnum),
                    hash: self.tcx.crate_hash(cnum),
                    kind: self.tcx.dep_kind(cnum),
                    extra_filename: self.tcx.extra_filename(cnum),
                };
                (cnum, dep)
            })
            .collect::<Vec<_>>();

        deps.sort_by_key(|&(cnum, _)| cnum);

        {
            // Sanity-check the crate numbers
            let mut expected_cnum = 1;
            for &(n, _) in &deps {
                assert_eq!(n, CrateNum::new(expected_cnum));
                expected_cnum += 1;
            }
        }

        // We're just going to write a list of crate 'name-hash-version's, with
        // the assumption that they are numbered 1 to n.
        // FIXME (#2166): This is not nearly enough to support correct versioning
        // but is enough to get transitive crate dependencies working.
        self.lazy_seq_ref(deps.iter().map(|&(_, ref dep)| dep))
    }

    fn encode_lib_features(&mut self) -> LazySeq<(ast::Name, Option<ast::Name>)> {
        let tcx = self.tcx;
        let lib_features = tcx.lib_features();
        self.lazy_seq(lib_features.to_vec())
    }

    fn encode_lang_items(&mut self) -> LazySeq<(DefIndex, usize)> {
        let tcx = self.tcx;
        let lang_items = tcx.lang_items();
        let lang_items = lang_items.items().iter();
        self.lazy_seq(lang_items.enumerate().filter_map(|(i, &opt_def_id)| {
            if let Some(def_id) = opt_def_id {
                if def_id.is_local() {
                    return Some((def_id.index, i));
                }
            }
            None
        }))
    }

    fn encode_lang_items_missing(&mut self) -> LazySeq<lang_items::LangItem> {
        let tcx = self.tcx;
        self.lazy_seq_ref(&tcx.lang_items().missing)
    }

    /// Encodes an index, mapping each trait to its (local) implementations.
    fn encode_impls(&mut self) -> LazySeq<TraitImpls> {
        debug!("EncodeContext::encode_impls()");
        let tcx = self.tcx;
        let mut visitor = ImplVisitor {
            tcx,
            impls: FxHashMap::default(),
        };
        tcx.hir().krate().visit_all_item_likes(&mut visitor);

        let mut all_impls: Vec<_> = visitor.impls.into_iter().collect();

        // Bring everything into deterministic order for hashing
        all_impls.sort_by_cached_key(|&(trait_def_id, _)| {
            tcx.def_path_hash(trait_def_id)
        });

        let all_impls: Vec<_> = all_impls
            .into_iter()
            .map(|(trait_def_id, mut impls)| {
                // Bring everything into deterministic order for hashing
                impls.sort_by_cached_key(|&def_index| {
                    tcx.hir().definitions().def_path_hash(def_index)
                });

                TraitImpls {
                    trait_id: (trait_def_id.krate.as_u32(), trait_def_id.index),
                    impls: self.lazy_seq_ref(&impls),
                }
            })
            .collect();

        self.lazy_seq_ref(&all_impls)
    }

    // Encodes all symbols exported from this crate into the metadata.
    //
    // This pass is seeded off the reachability list calculated in the
    // middle::reachable module but filters out items that either don't have a
    // symbol associated with them (they weren't translated) or if they're an FFI
    // definition (as that's not defined in this crate).
    fn encode_exported_symbols(&mut self,
                               exported_symbols: &[(ExportedSymbol<'tcx>, SymbolExportLevel)])
                               -> LazySeq<(ExportedSymbol<'tcx>, SymbolExportLevel)> {
        // The metadata symbol name is special. It should not show up in
        // downstream crates.
        let metadata_symbol_name = SymbolName::new(&metadata_symbol_name(self.tcx));

        self.lazy_seq(exported_symbols
            .iter()
            .filter(|&&(ref exported_symbol, _)| {
                match *exported_symbol {
                    ExportedSymbol::NoDefId(symbol_name) => {
                        symbol_name != metadata_symbol_name
                    },
                    _ => true,
                }
            })
            .cloned())
    }

    fn encode_dylib_dependency_formats(&mut self) -> LazySeq<Option<LinkagePreference>> {
        match self.tcx.sess.dependency_formats.borrow().get(&config::CrateType::Dylib) {
            Some(arr) => {
                self.lazy_seq(arr.iter().map(|slot| {
                    match *slot {
                        Linkage::NotLinked |
                        Linkage::IncludedFromDylib => None,

                        Linkage::Dynamic => Some(LinkagePreference::RequireDynamic),
                        Linkage::Static => Some(LinkagePreference::RequireStatic),
                    }
                }))
            }
            None => LazySeq::empty(),
        }
    }

    fn encode_info_for_foreign_item(&mut self,
                                    (def_id, nitem): (DefId, &hir::ForeignItem))
                                    -> Entry<'tcx> {
        let tcx = self.tcx;

        debug!("EncodeContext::encode_info_for_foreign_item({:?})", def_id);

        let kind = match nitem.node {
            hir::ForeignItemKind::Fn(_, ref names, _) => {
                let data = FnData {
                    constness: hir::Constness::NotConst,
                    arg_names: self.encode_fn_arg_names(names),
                    sig: self.lazy(&tcx.fn_sig(def_id)),
                };
                EntryKind::ForeignFn(self.lazy(&data))
            }
            hir::ForeignItemKind::Static(_, hir::MutMutable) => EntryKind::ForeignMutStatic,
            hir::ForeignItemKind::Static(_, hir::MutImmutable) => EntryKind::ForeignImmStatic,
            hir::ForeignItemKind::Type => EntryKind::ForeignType,
        };

        Entry {
            kind,
            visibility: self.lazy(&ty::Visibility::from_hir(&nitem.vis, nitem.hir_id, tcx)),
            span: self.lazy(&nitem.span),
            attributes: self.encode_attributes(&nitem.attrs),
            children: LazySeq::empty(),
            stability: self.encode_stability(def_id),
            deprecation: self.encode_deprecation(def_id),

            ty: Some(self.encode_item_type(def_id)),
            inherent_impls: LazySeq::empty(),
            variances: match nitem.node {
                hir::ForeignItemKind::Fn(..) => self.encode_variances_of(def_id),
                _ => LazySeq::empty(),
            },
            generics: Some(self.encode_generics(def_id)),
            predicates: Some(self.encode_predicates(def_id)),
            predicates_defined_on: None,

            mir: None,
        }
    }
}

impl Visitor<'tcx> for EncodeContext<'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.tcx.hir())
    }
    fn visit_expr(&mut self, ex: &'tcx hir::Expr) {
        intravisit::walk_expr(self, ex);
        self.encode_info_for_expr(ex);
    }
    fn visit_item(&mut self, item: &'tcx hir::Item) {
        intravisit::walk_item(self, item);
        let def_id = self.tcx.hir().local_def_id_from_hir_id(item.hir_id);
        match item.node {
            hir::ItemKind::ExternCrate(_) |
            hir::ItemKind::Use(..) => {} // ignore these
            _ => self.record(def_id, EncodeContext::encode_info_for_item, (def_id, item)),
        }
        self.encode_addl_info_for_item(item);
    }
    fn visit_foreign_item(&mut self, ni: &'tcx hir::ForeignItem) {
        intravisit::walk_foreign_item(self, ni);
        let def_id = self.tcx.hir().local_def_id_from_hir_id(ni.hir_id);
        self.record(def_id,
                          EncodeContext::encode_info_for_foreign_item,
                          (def_id, ni));
    }
    fn visit_variant(&mut self,
                     v: &'tcx hir::Variant,
                     g: &'tcx hir::Generics,
                     id: hir::HirId) {
        intravisit::walk_variant(self, v, g, id);

        if let Some(ref discr) = v.node.disr_expr {
            let def_id = self.tcx.hir().local_def_id_from_hir_id(discr.hir_id);
            self.record(def_id, EncodeContext::encode_info_for_anon_const, def_id);
        }
    }
    fn visit_generics(&mut self, generics: &'tcx hir::Generics) {
        intravisit::walk_generics(self, generics);
        self.encode_info_for_generics(generics);
    }
    fn visit_ty(&mut self, ty: &'tcx hir::Ty) {
        intravisit::walk_ty(self, ty);
        self.encode_info_for_ty(ty);
    }
    fn visit_macro_def(&mut self, macro_def: &'tcx hir::MacroDef) {
        let def_id = self.tcx.hir().local_def_id_from_hir_id(macro_def.hir_id);
        self.record(def_id, EncodeContext::encode_info_for_macro_def, macro_def);
    }
}

impl EncodeContext<'tcx> {
    fn encode_fields(&mut self, adt_def_id: DefId) {
        let def = self.tcx.adt_def(adt_def_id);
        for (variant_index, variant) in def.variants.iter_enumerated() {
            for (field_index, field) in variant.fields.iter().enumerate() {
                self.record(field.did,
                            EncodeContext::encode_field,
                            (adt_def_id, variant_index, field_index));
            }
        }
    }

    fn encode_info_for_generics(&mut self, generics: &hir::Generics) {
        for param in &generics.params {
            let def_id = self.tcx.hir().local_def_id_from_hir_id(param.hir_id);
            match param.kind {
                GenericParamKind::Lifetime { .. } => continue,
                GenericParamKind::Type { ref default, .. } => {
                    self.record(
                        def_id,
                        EncodeContext::encode_info_for_ty_param,
                        (def_id, default.is_some()),
                    );
                }
                GenericParamKind::Const { .. } => {
                    self.record(def_id, EncodeContext::encode_info_for_const_param, def_id);
                }
            }
        }
    }

    fn encode_info_for_ty(&mut self, ty: &hir::Ty) {
        match ty.node {
            hir::TyKind::Array(_, ref length) => {
                let def_id = self.tcx.hir().local_def_id_from_hir_id(length.hir_id);
                self.record(def_id, EncodeContext::encode_info_for_anon_const, def_id);
            }
            _ => {}
        }
    }

    fn encode_info_for_expr(&mut self, expr: &hir::Expr) {
        match expr.node {
            hir::ExprKind::Closure(..) => {
                let def_id = self.tcx.hir().local_def_id_from_hir_id(expr.hir_id);
                self.record(def_id, EncodeContext::encode_info_for_closure, def_id);
            }
            _ => {}
        }
    }

    /// In some cases, along with the item itself, we also
    /// encode some sub-items. Usually we want some info from the item
    /// so it's easier to do that here then to wait until we would encounter
    /// normally in the visitor walk.
    fn encode_addl_info_for_item(&mut self, item: &hir::Item) {
        let def_id = self.tcx.hir().local_def_id_from_hir_id(item.hir_id);
        match item.node {
            hir::ItemKind::Static(..) |
            hir::ItemKind::Const(..) |
            hir::ItemKind::Fn(..) |
            hir::ItemKind::Mod(..) |
            hir::ItemKind::ForeignMod(..) |
            hir::ItemKind::GlobalAsm(..) |
            hir::ItemKind::ExternCrate(..) |
            hir::ItemKind::Use(..) |
            hir::ItemKind::Ty(..) |
            hir::ItemKind::Existential(..) |
            hir::ItemKind::TraitAlias(..) => {
                // no sub-item recording needed in these cases
            }
            hir::ItemKind::Enum(..) => {
                self.encode_fields(def_id);

                let def = self.tcx.adt_def(def_id);
                for (i, variant) in def.variants.iter_enumerated() {
                    self.record(variant.def_id,
                                EncodeContext::encode_enum_variant_info,
                                (def_id, i));

                    if let Some(ctor_def_id) = variant.ctor_def_id {
                        self.record(ctor_def_id,
                                    EncodeContext::encode_enum_variant_ctor,
                                    (def_id, i));
                    }
                }
            }
            hir::ItemKind::Struct(ref struct_def, _) => {
                self.encode_fields(def_id);

                // If the struct has a constructor, encode it.
                if let Some(ctor_hir_id) = struct_def.ctor_hir_id() {
                    let ctor_def_id = self.tcx.hir().local_def_id_from_hir_id(ctor_hir_id);
                    self.record(ctor_def_id,
                                EncodeContext::encode_struct_ctor,
                                (def_id, ctor_def_id));
                }
            }
            hir::ItemKind::Union(..) => {
                self.encode_fields(def_id);
            }
            hir::ItemKind::Impl(..) => {
                for &trait_item_def_id in self.tcx.associated_item_def_ids(def_id).iter() {
                    self.record(trait_item_def_id,
                                EncodeContext::encode_info_for_impl_item,
                                trait_item_def_id);
                }
            }
            hir::ItemKind::Trait(..) => {
                for &item_def_id in self.tcx.associated_item_def_ids(def_id).iter() {
                    self.record(item_def_id,
                                EncodeContext::encode_info_for_trait_item,
                                item_def_id);
                }
            }
        }
    }
}

struct ImplVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    impls: FxHashMap<DefId, Vec<DefIndex>>,
}

impl<'tcx, 'v> ItemLikeVisitor<'v> for ImplVisitor<'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        if let hir::ItemKind::Impl(..) = item.node {
            let impl_id = self.tcx.hir().local_def_id_from_hir_id(item.hir_id);
            if let Some(trait_ref) = self.tcx.impl_trait_ref(impl_id) {
                self.impls
                    .entry(trait_ref.def_id)
                    .or_default()
                    .push(impl_id.index);
            }
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &'v hir::TraitItem) {}

    fn visit_impl_item(&mut self, _impl_item: &'v hir::ImplItem) {
        // handled in `visit_item` above
    }
}

// NOTE(eddyb) The following comment was preserved for posterity, even
// though it's no longer relevant as EBML (which uses nested & tagged
// "documents") was replaced with a scheme that can't go out of bounds.
//
// And here we run into yet another obscure archive bug: in which metadata
// loaded from archives may have trailing garbage bytes. Awhile back one of
// our tests was failing sporadically on the macOS 64-bit builders (both nopt
// and opt) by having ebml generate an out-of-bounds panic when looking at
// metadata.
//
// Upon investigation it turned out that the metadata file inside of an rlib
// (and ar archive) was being corrupted. Some compilations would generate a
// metadata file which would end in a few extra bytes, while other
// compilations would not have these extra bytes appended to the end. These
// extra bytes were interpreted by ebml as an extra tag, so they ended up
// being interpreted causing the out-of-bounds.
//
// The root cause of why these extra bytes were appearing was never
// discovered, and in the meantime the solution we're employing is to insert
// the length of the metadata to the start of the metadata. Later on this
// will allow us to slice the metadata to the precise length that we just
// generated regardless of trailing bytes that end up in it.

pub fn encode_metadata(tcx: TyCtxt<'_>) -> EncodedMetadata {
    let mut encoder = opaque::Encoder::new(vec![]);
    encoder.emit_raw_bytes(METADATA_HEADER);

    // Will be filled with the root position after encoding everything.
    encoder.emit_raw_bytes(&[0, 0, 0, 0]);

    // Since encoding metadata is not in a query, and nothing is cached,
    // there's no need to do dep-graph tracking for any of it.
    let (root, mut result) = tcx.dep_graph.with_ignore(move || {
        let mut ecx = EncodeContext {
            opaque: encoder,
            tcx,
            entries_index: Index::new(tcx.hir().definitions().def_index_count()),
            lazy_state: LazyState::NoNode,
            type_shorthands: Default::default(),
            predicate_shorthands: Default::default(),
            source_file_cache: tcx.sess.source_map().files()[0].clone(),
            interpret_allocs: Default::default(),
            interpret_allocs_inverse: Default::default(),
        };

        // Encode the rustc version string in a predictable location.
        rustc_version().encode(&mut ecx).unwrap();

        // Encode all the entries and extra information in the crate,
        // culminating in the `CrateRoot` which points to all of it.
        let root = ecx.encode_crate_root();
        (root, ecx.opaque.into_inner())
    });

    // Encode the root position.
    let header = METADATA_HEADER.len();
    let pos = root.position;
    result[header + 0] = (pos >> 24) as u8;
    result[header + 1] = (pos >> 16) as u8;
    result[header + 2] = (pos >> 8) as u8;
    result[header + 3] = (pos >> 0) as u8;

    EncodedMetadata { raw_data: result }
}

pub fn get_repr_options(tcx: TyCtxt<'_>, did: DefId) -> ReprOptions {
    let ty = tcx.type_of(did);
    match ty.sty {
        ty::Adt(ref def, _) => return def.repr,
        _ => bug!("{} is not an ADT", ty),
    }
}
