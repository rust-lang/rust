// The crate store - a central repo for information collected about external
// crates and libraries
use crate::schema::{self, *};
use log::debug;
use proc_macro::bridge::client::ProcMacro;
use rustc::dep_graph::{self, DepNodeIndex};
use rustc::hir;
use rustc::hir::def::{self, Res, DefKind, CtorOf, CtorKind};
use rustc::hir::def_id::{CrateNum, DefId, DefIndex, CRATE_DEF_INDEX};
use rustc::hir::map::{DefKey, DefPath, DefPathData, DefPathHash};
use rustc::hir::map::definitions::DefPathTable;
use rustc::middle::lang_items;
use rustc::middle::cstore::{
    CrateSource, DepKind, ExternCrate, ForeignModule, LinkagePreference, NativeLibrary
};
use rustc::middle::exported_symbols::{ExportedSymbol, SymbolExportLevel};
use rustc::mir::{Body, Promoted};
use rustc::mir::interpret::AllocDecodingState;
use rustc::util::nodemap::FxHashMap;
use rustc::session::Session;
use rustc::ty::{self, Ty, TyCtxt};
use rustc_data_structures::sync::{Lrc, Lock, MetadataRef, Once, AtomicCell};
use rustc_data_structures::svh::Svh;
use rustc_index::vec::IndexVec;
use syntax::ast::{self, Ident};
use syntax::attr;
use syntax::edition::Edition;
use syntax::source_map::{self, respan, Spanned};
use syntax_expand::base::{SyntaxExtensionKind, SyntaxExtension};
use syntax_expand::proc_macro::{AttrProcMacro, ProcMacroDerive, BangProcMacro};
use syntax_pos::{self, Span, Pos, DUMMY_SP, hygiene::MacroKind};
use syntax_pos::symbol::{Symbol, sym};

pub use crate::cstore_impl::{provide, provide_extern};

// A map from external crate numbers (as decoded from some crate file) to
// local crate numbers (as generated during this session). Each external
// crate may refer to types in other external crates, and each has their
// own crate numbers.
crate type CrateNumMap = IndexVec<CrateNum, CrateNum>;

crate struct MetadataBlob(pub MetadataRef);

/// Holds information about a syntax_pos::SourceFile imported from another crate.
/// See `imported_source_files()` for more information.
crate struct ImportedSourceFile {
    /// This SourceFile's byte-offset within the source_map of its original crate
    pub original_start_pos: syntax_pos::BytePos,
    /// The end of this SourceFile within the source_map of its original crate
    pub original_end_pos: syntax_pos::BytePos,
    /// The imported SourceFile's representation within the local source_map
    pub translated_source_file: Lrc<syntax_pos::SourceFile>,
}

crate struct CrateMetadata {
    /// The primary crate data - binary metadata blob.
    crate blob: MetadataBlob,

    // --- Some data pre-decoded from the metadata blob, usually for performance ---

    /// Properties of the whole crate.
    /// NOTE(eddyb) we pass `'static` to a `'tcx` parameter because this
    /// lifetime is only used behind `Lazy`, and therefore acts like an
    /// universal (`for<'tcx>`), that is paired up with whichever `TyCtxt`
    /// is being used to decode those values.
    crate root: schema::CrateRoot<'static>,
    /// For each definition in this crate, we encode a key. When the
    /// crate is loaded, we read all the keys and put them in this
    /// hashmap, which gives the reverse mapping. This allows us to
    /// quickly retrace a `DefPath`, which is needed for incremental
    /// compilation support.
    crate def_path_table: DefPathTable,
    /// Trait impl data.
    /// FIXME: Used only from queries and can use query cache,
    /// so pre-decoding can probably be avoided.
    crate trait_impls: FxHashMap<(u32, DefIndex), schema::Lazy<[DefIndex]>>,
    /// Proc macro descriptions for this crate, if it's a proc macro crate.
    crate raw_proc_macros: Option<&'static [ProcMacro]>,
    /// Source maps for code from the crate.
    crate source_map_import_info: Once<Vec<ImportedSourceFile>>,
    /// Used for decoding interpret::AllocIds in a cached & thread-safe manner.
    crate alloc_decoding_state: AllocDecodingState,
    /// The `DepNodeIndex` of the `DepNode` representing this upstream crate.
    /// It is initialized on the first access in `get_crate_dep_node_index()`.
    /// Do not access the value directly, as it might not have been initialized yet.
    /// The field must always be initialized to `DepNodeIndex::INVALID`.
    crate dep_node_index: AtomicCell<DepNodeIndex>,

    // --- Other significant crate properties ---

    /// ID of this crate, from the current compilation session's point of view.
    crate cnum: CrateNum,
    /// Maps crate IDs as they are were seen from this crate's compilation sessions into
    /// IDs as they are seen from the current compilation session.
    crate cnum_map: CrateNumMap,
    /// Same ID set as `cnum_map` plus maybe some injected crates like panic runtime.
    crate dependencies: Lock<Vec<CrateNum>>,
    /// How to link (or not link) this crate to the currently compiled crate.
    crate dep_kind: Lock<DepKind>,
    /// Filesystem location of this crate.
    crate source: CrateSource,
    /// Whether or not this crate should be consider a private dependency
    /// for purposes of the 'exported_private_dependencies' lint
    crate private_dep: bool,
    /// The hash for the host proc macro. Used to support `-Z dual-proc-macro`.
    crate host_hash: Option<Svh>,

    // --- Data used only for improving diagnostics ---

    /// Information about the `extern crate` item or path that caused this crate to be loaded.
    /// If this is `None`, then the crate was injected (e.g., by the allocator).
    crate extern_crate: Lock<Option<ExternCrate>>,
}

impl<'a, 'tcx> CrateMetadata {
    crate fn is_proc_macro_crate(&self) -> bool {
        self.root.proc_macro_decls_static.is_some()
    }

    fn is_proc_macro(&self, id: DefIndex) -> bool {
        self.is_proc_macro_crate() &&
            self.root.proc_macro_data.unwrap().decode(self).find(|x| *x == id).is_some()
    }

    fn maybe_kind(&self, item_id: DefIndex) -> Option<EntryKind<'tcx>> {
        self.root.per_def.kind.get(self, item_id).map(|k| k.decode(self))
    }

    fn kind(&self, item_id: DefIndex) -> EntryKind<'tcx> {
        assert!(!self.is_proc_macro(item_id));
        self.maybe_kind(item_id).unwrap_or_else(|| {
            bug!(
                "CrateMetadata::kind({:?}): id not found, in crate {:?} with number {}",
                item_id,
                self.root.name,
                self.cnum,
            )
        })
    }

    fn local_def_id(&self, index: DefIndex) -> DefId {
        DefId {
            krate: self.cnum,
            index,
        }
    }

    fn raw_proc_macro(&self, id: DefIndex) -> &ProcMacro {
        // DefIndex's in root.proc_macro_data have a one-to-one correspondence
        // with items in 'raw_proc_macros'.
        // NOTE: If you update the order of macros in 'proc_macro_data' for any reason,
        // you must also update src/libsyntax_ext/proc_macro_harness.rs
        // Failing to do so will result in incorrect data being associated
        // with proc macros when deserialized.
        let pos = self.root.proc_macro_data.unwrap().decode(self).position(|i| i == id).unwrap();
        &self.raw_proc_macros.unwrap()[pos]
    }

    crate fn item_name(&self, item_index: DefIndex) -> Symbol {
        if !self.is_proc_macro(item_index) {
            self.def_key(item_index)
                .disambiguated_data
                .data
                .get_opt_name()
                .expect("no name in item_name")
        } else {
            Symbol::intern(self.raw_proc_macro(item_index).name())
        }
    }

    crate fn def_kind(&self, index: DefIndex) -> Option<DefKind> {
        if !self.is_proc_macro(index) {
            self.kind(index).def_kind()
        } else {
            Some(DefKind::Macro(
                macro_kind(self.raw_proc_macro(index))
            ))
        }
    }

    crate fn get_span(&self, index: DefIndex, sess: &Session) -> Span {
        self.root.per_def.span.get(self, index).unwrap().decode((self, sess))
    }

    crate fn load_proc_macro(&self, id: DefIndex, sess: &Session) -> SyntaxExtension {
        let (name, kind, helper_attrs) = match *self.raw_proc_macro(id) {
            ProcMacro::CustomDerive { trait_name, attributes, client } => {
                let helper_attrs =
                    attributes.iter().cloned().map(Symbol::intern).collect::<Vec<_>>();
                (
                    trait_name,
                    SyntaxExtensionKind::Derive(Box::new(ProcMacroDerive { client })),
                    helper_attrs,
                )
            }
            ProcMacro::Attr { name, client } => (
                name, SyntaxExtensionKind::Attr(Box::new(AttrProcMacro { client })), Vec::new()
            ),
            ProcMacro::Bang { name, client } => (
                name, SyntaxExtensionKind::Bang(Box::new(BangProcMacro { client })), Vec::new()
            )
        };

        SyntaxExtension::new(
            &sess.parse_sess,
            kind,
            self.get_span(id, sess),
            helper_attrs,
            self.root.edition,
            Symbol::intern(name),
            &self.get_item_attrs(id, sess),
        )
    }

    crate fn get_trait_def(&self, item_id: DefIndex, sess: &Session) -> ty::TraitDef {
        match self.kind(item_id) {
            EntryKind::Trait(data) => {
                let data = data.decode((self, sess));
                ty::TraitDef::new(self.local_def_id(item_id),
                                  data.unsafety,
                                  data.paren_sugar,
                                  data.has_auto_impl,
                                  data.is_marker,
                                  self.def_path_table.def_path_hash(item_id))
            },
            EntryKind::TraitAlias => {
                ty::TraitDef::new(self.local_def_id(item_id),
                                  hir::Unsafety::Normal,
                                  false,
                                  false,
                                  false,
                                  self.def_path_table.def_path_hash(item_id))
            },
            _ => bug!("def-index does not refer to trait or trait alias"),
        }
    }

    fn get_variant(
        &self,
        tcx: TyCtxt<'tcx>,
        kind: &EntryKind<'_>,
        index: DefIndex,
        parent_did: DefId,
    ) -> ty::VariantDef {
        let data = match kind {
            EntryKind::Variant(data) |
            EntryKind::Struct(data, _) |
            EntryKind::Union(data, _) => data.decode(self),
            _ => bug!(),
        };

        let adt_kind = match kind {
            EntryKind::Variant(_) => ty::AdtKind::Enum,
            EntryKind::Struct(..) => ty::AdtKind::Struct,
            EntryKind::Union(..) => ty::AdtKind::Union,
            _ => bug!(),
        };

        let variant_did = if adt_kind == ty::AdtKind::Enum {
            Some(self.local_def_id(index))
        } else {
            None
        };
        let ctor_did = data.ctor.map(|index| self.local_def_id(index));

        ty::VariantDef::new(
            tcx,
            Ident::with_dummy_span(self.item_name(index)),
            variant_did,
            ctor_did,
            data.discr,
            self.root.per_def.children.get(self, index).unwrap_or(Lazy::empty())
                .decode(self).map(|index| ty::FieldDef {
                    did: self.local_def_id(index),
                    ident: Ident::with_dummy_span(self.item_name(index)),
                    vis: self.get_visibility(index),
                }).collect(),
            data.ctor_kind,
            adt_kind,
            parent_did,
            false,
        )
    }

    crate fn get_adt_def(&self, item_id: DefIndex, tcx: TyCtxt<'tcx>) -> &'tcx ty::AdtDef {
        let kind = self.kind(item_id);
        let did = self.local_def_id(item_id);

        let (adt_kind, repr) = match kind {
            EntryKind::Enum(repr) => (ty::AdtKind::Enum, repr),
            EntryKind::Struct(_, repr) => (ty::AdtKind::Struct, repr),
            EntryKind::Union(_, repr) => (ty::AdtKind::Union, repr),
            _ => bug!("get_adt_def called on a non-ADT {:?}", did),
        };

        let variants = if let ty::AdtKind::Enum = adt_kind {
            self.root.per_def.children.get(self, item_id).unwrap_or(Lazy::empty())
                .decode(self)
                .map(|index| {
                    self.get_variant(tcx, &self.kind(index), index, did)
                })
                .collect()
        } else {
            std::iter::once(self.get_variant(tcx, &kind, item_id, did)).collect()
        };

        tcx.alloc_adt_def(did, adt_kind, variants, repr)
    }

    crate fn get_predicates(
        &self,
        item_id: DefIndex,
        tcx: TyCtxt<'tcx>,
    ) -> ty::GenericPredicates<'tcx> {
        self.root.per_def.predicates.get(self, item_id).unwrap().decode((self, tcx))
    }

    crate fn get_predicates_defined_on(
        &self,
        item_id: DefIndex,
        tcx: TyCtxt<'tcx>,
    ) -> ty::GenericPredicates<'tcx> {
        self.root.per_def.predicates_defined_on.get(self, item_id).unwrap().decode((self, tcx))
    }

    crate fn get_super_predicates(
        &self,
        item_id: DefIndex,
        tcx: TyCtxt<'tcx>,
    ) -> ty::GenericPredicates<'tcx> {
        self.root.per_def.super_predicates.get(self, item_id).unwrap().decode((self, tcx))
    }

    crate fn get_generics(&self, item_id: DefIndex, sess: &Session) -> ty::Generics {
        self.root.per_def.generics.get(self, item_id).unwrap().decode((self, sess))
    }

    crate fn get_type(&self, id: DefIndex, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        self.root.per_def.ty.get(self, id).unwrap().decode((self, tcx))
    }

    crate fn get_stability(&self, id: DefIndex) -> Option<attr::Stability> {
        match self.is_proc_macro(id) {
            true => self.root.proc_macro_stability.clone(),
            false => self.root.per_def.stability.get(self, id).map(|stab| stab.decode(self)),
        }
    }

    crate fn get_deprecation(&self, id: DefIndex) -> Option<attr::Deprecation> {
        self.root.per_def.deprecation.get(self, id)
            .filter(|_| !self.is_proc_macro(id))
            .map(|depr| depr.decode(self))
    }

    crate fn get_visibility(&self, id: DefIndex) -> ty::Visibility {
        match self.is_proc_macro(id) {
            true => ty::Visibility::Public,
            false => self.root.per_def.visibility.get(self, id).unwrap().decode(self),
        }
    }

    fn get_impl_data(&self, id: DefIndex) -> ImplData {
        match self.kind(id) {
            EntryKind::Impl(data) => data.decode(self),
            _ => bug!(),
        }
    }

    crate fn get_parent_impl(&self, id: DefIndex) -> Option<DefId> {
        self.get_impl_data(id).parent_impl
    }

    crate fn get_impl_polarity(&self, id: DefIndex) -> ty::ImplPolarity {
        self.get_impl_data(id).polarity
    }

    crate fn get_impl_defaultness(&self, id: DefIndex) -> hir::Defaultness {
        self.get_impl_data(id).defaultness
    }

    crate fn get_coerce_unsized_info(
        &self,
        id: DefIndex,
    ) -> Option<ty::adjustment::CoerceUnsizedInfo> {
        self.get_impl_data(id).coerce_unsized_info
    }

    crate fn get_impl_trait(&self, id: DefIndex, tcx: TyCtxt<'tcx>) -> Option<ty::TraitRef<'tcx>> {
        self.root.per_def.impl_trait_ref.get(self, id).map(|tr| tr.decode((self, tcx)))
    }

    /// Iterates over all the stability attributes in the given crate.
    crate fn get_lib_features(&self, tcx: TyCtxt<'tcx>) -> &'tcx [(ast::Name, Option<ast::Name>)] {
        // FIXME: For a proc macro crate, not sure whether we should return the "host"
        // features or an empty Vec. Both don't cause ICEs.
        tcx.arena.alloc_from_iter(self.root
            .lib_features
            .decode(self))
    }

    /// Iterates over the language items in the given crate.
    crate fn get_lang_items(&self, tcx: TyCtxt<'tcx>) -> &'tcx [(DefId, usize)] {
        if self.is_proc_macro_crate() {
            // Proc macro crates do not export any lang-items to the target.
            &[]
        } else {
            tcx.arena.alloc_from_iter(self.root
                .lang_items
                .decode(self)
                .map(|(def_index, index)| (self.local_def_id(def_index), index)))
        }
    }

    /// Iterates over the diagnostic items in the given crate.
    crate fn get_diagnostic_items(
        &self,
        tcx: TyCtxt<'tcx>,
    ) -> &'tcx FxHashMap<Symbol, DefId> {
        tcx.arena.alloc(if self.is_proc_macro_crate() {
            // Proc macro crates do not export any diagnostic-items to the target.
            Default::default()
        } else {
            self.root
                .diagnostic_items
                .decode(self)
                .map(|(name, def_index)| (name, self.local_def_id(def_index)))
                .collect()
        })
    }

    /// Iterates over each child of the given item.
    crate fn each_child_of_item<F>(&self, id: DefIndex, mut callback: F, sess: &Session)
        where F: FnMut(def::Export<hir::HirId>)
    {
        if let Some(proc_macros_ids) = self.root.proc_macro_data.map(|d| d.decode(self)) {
            /* If we are loading as a proc macro, we want to return the view of this crate
             * as a proc macro crate.
             */
            if id == CRATE_DEF_INDEX {
                for def_index in proc_macros_ids {
                    let raw_macro = self.raw_proc_macro(def_index);
                    let res = Res::Def(
                        DefKind::Macro(macro_kind(raw_macro)),
                        self.local_def_id(def_index),
                    );
                    let ident = Ident::from_str(raw_macro.name());
                    callback(def::Export {
                        ident: ident,
                        res: res,
                        vis: ty::Visibility::Public,
                        span: DUMMY_SP,
                    });
                }
            }
            return
        }

        // Find the item.
        let kind = match self.maybe_kind(id) {
            None => return,
            Some(kind) => kind,
        };

        // Iterate over all children.
        let macros_only = self.dep_kind.lock().macros_only();
        let children = self.root.per_def.children.get(self, id).unwrap_or(Lazy::empty());
        for child_index in children.decode((self, sess)) {
            if macros_only {
                continue
            }

            // Get the item.
            if let Some(child_kind) = self.maybe_kind(child_index) {
                match child_kind {
                    EntryKind::MacroDef(..) => {}
                    _ if macros_only => continue,
                    _ => {}
                }

                // Hand off the item to the callback.
                match child_kind {
                    // FIXME(eddyb) Don't encode these in children.
                    EntryKind::ForeignMod => {
                        let child_children =
                            self.root.per_def.children.get(self, child_index)
                                .unwrap_or(Lazy::empty());
                        for child_index in child_children.decode((self, sess)) {
                            if let Some(kind) = self.def_kind(child_index) {
                                callback(def::Export {
                                    res: Res::Def(kind, self.local_def_id(child_index)),
                                    ident: Ident::with_dummy_span(self.item_name(child_index)),
                                    vis: self.get_visibility(child_index),
                                    span: self.root.per_def.span.get(self, child_index).unwrap()
                                        .decode((self, sess)),
                                });
                            }
                        }
                        continue;
                    }
                    EntryKind::Impl(_) => continue,

                    _ => {}
                }

                let def_key = self.def_key(child_index);
                let span = self.get_span(child_index, sess);
                if let (Some(kind), Some(name)) =
                    (self.def_kind(child_index), def_key.disambiguated_data.data.get_opt_name()) {
                    let ident = Ident::with_dummy_span(name);
                    let vis = self.get_visibility(child_index);
                    let def_id = self.local_def_id(child_index);
                    let res = Res::Def(kind, def_id);
                    callback(def::Export { res, ident, vis, span });
                    // For non-re-export structs and variants add their constructors to children.
                    // Re-export lists automatically contain constructors when necessary.
                    match kind {
                        DefKind::Struct => {
                            if let Some(ctor_def_id) = self.get_ctor_def_id(child_index) {
                                let ctor_kind = self.get_ctor_kind(child_index);
                                let ctor_res = Res::Def(
                                    DefKind::Ctor(CtorOf::Struct, ctor_kind),
                                    ctor_def_id,
                                );
                                let vis = self.get_visibility(ctor_def_id.index);
                                callback(def::Export { res: ctor_res, vis, ident, span });
                            }
                        }
                        DefKind::Variant => {
                            // Braced variants, unlike structs, generate unusable names in
                            // value namespace, they are reserved for possible future use.
                            // It's ok to use the variant's id as a ctor id since an
                            // error will be reported on any use of such resolution anyway.
                            let ctor_def_id = self.get_ctor_def_id(child_index).unwrap_or(def_id);
                            let ctor_kind = self.get_ctor_kind(child_index);
                            let ctor_res = Res::Def(
                                DefKind::Ctor(CtorOf::Variant, ctor_kind),
                                ctor_def_id,
                            );
                            let mut vis = self.get_visibility(ctor_def_id.index);
                            if ctor_def_id == def_id && vis == ty::Visibility::Public {
                                // For non-exhaustive variants lower the constructor visibility to
                                // within the crate. We only need this for fictive constructors,
                                // for other constructors correct visibilities
                                // were already encoded in metadata.
                                let attrs = self.get_item_attrs(def_id.index, sess);
                                if attr::contains_name(&attrs, sym::non_exhaustive) {
                                    let crate_def_id = self.local_def_id(CRATE_DEF_INDEX);
                                    vis = ty::Visibility::Restricted(crate_def_id);
                                }
                            }
                            callback(def::Export { res: ctor_res, ident, vis, span });
                        }
                        _ => {}
                    }
                }
            }
        }

        if let EntryKind::Mod(data) = kind {
            for exp in data.decode((self, sess)).reexports.decode((self, sess)) {
                match exp.res {
                    Res::Def(DefKind::Macro(..), _) => {}
                    _ if macros_only => continue,
                    _ => {}
                }
                callback(exp);
            }
        }
    }

    crate fn is_item_mir_available(&self, id: DefIndex) -> bool {
        !self.is_proc_macro(id) &&
            self.root.per_def.mir.get(self, id).is_some()
    }

    crate fn get_optimized_mir(&self, tcx: TyCtxt<'tcx>, id: DefIndex) -> Body<'tcx> {
        self.root.per_def.mir.get(self, id)
            .filter(|_| !self.is_proc_macro(id))
            .unwrap_or_else(|| {
                bug!("get_optimized_mir: missing MIR for `{:?}`", self.local_def_id(id))
            })
            .decode((self, tcx))
    }

    crate fn get_promoted_mir(
        &self,
        tcx: TyCtxt<'tcx>,
        id: DefIndex,
    ) -> IndexVec<Promoted, Body<'tcx>> {
        self.root.per_def.promoted_mir.get(self, id)
            .filter(|_| !self.is_proc_macro(id))
            .unwrap_or_else(|| {
                bug!("get_promoted_mir: missing MIR for `{:?}`", self.local_def_id(id))
            })
            .decode((self, tcx))
    }

    crate fn mir_const_qualif(&self, id: DefIndex) -> u8 {
        match self.kind(id) {
            EntryKind::Const(qualif, _) |
            EntryKind::AssocConst(AssocContainer::ImplDefault, qualif, _) |
            EntryKind::AssocConst(AssocContainer::ImplFinal, qualif, _) => {
                qualif.mir
            }
            _ => bug!(),
        }
    }

    crate fn get_associated_item(&self, id: DefIndex) -> ty::AssocItem {
        let def_key = self.def_key(id);
        let parent = self.local_def_id(def_key.parent.unwrap());
        let name = def_key.disambiguated_data.data.get_opt_name().unwrap();

        let (kind, container, has_self) = match self.kind(id) {
            EntryKind::AssocConst(container, _, _) => {
                (ty::AssocKind::Const, container, false)
            }
            EntryKind::Method(data) => {
                let data = data.decode(self);
                (ty::AssocKind::Method, data.container, data.has_self)
            }
            EntryKind::AssocType(container) => {
                (ty::AssocKind::Type, container, false)
            }
            EntryKind::AssocOpaqueTy(container) => {
                (ty::AssocKind::OpaqueTy, container, false)
            }
            _ => bug!("cannot get associated-item of `{:?}`", def_key)
        };

        ty::AssocItem {
            ident: Ident::with_dummy_span(name),
            kind,
            vis: self.get_visibility(id),
            defaultness: container.defaultness(),
            def_id: self.local_def_id(id),
            container: container.with_def_id(parent),
            method_has_self_argument: has_self
        }
    }

    crate fn get_item_variances(&self, id: DefIndex) -> Vec<ty::Variance> {
        self.root.per_def.variances.get(self, id).unwrap_or(Lazy::empty())
            .decode(self).collect()
    }

    crate fn get_ctor_kind(&self, node_id: DefIndex) -> CtorKind {
        match self.kind(node_id) {
            EntryKind::Struct(data, _) |
            EntryKind::Union(data, _) |
            EntryKind::Variant(data) => data.decode(self).ctor_kind,
            _ => CtorKind::Fictive,
        }
    }

    crate fn get_ctor_def_id(&self, node_id: DefIndex) -> Option<DefId> {
        match self.kind(node_id) {
            EntryKind::Struct(data, _) => {
                data.decode(self).ctor.map(|index| self.local_def_id(index))
            }
            EntryKind::Variant(data) => {
                data.decode(self).ctor.map(|index| self.local_def_id(index))
            }
            _ => None,
        }
    }

    crate fn get_item_attrs(&self, node_id: DefIndex, sess: &Session) -> Lrc<[ast::Attribute]> {
        // The attributes for a tuple struct/variant are attached to the definition, not the ctor;
        // we assume that someone passing in a tuple struct ctor is actually wanting to
        // look at the definition
        let def_key = self.def_key(node_id);
        let item_id = if def_key.disambiguated_data.data == DefPathData::Ctor {
            def_key.parent.unwrap()
        } else {
            node_id
        };

        Lrc::from(self.root.per_def.attributes.get(self, item_id).unwrap_or(Lazy::empty())
            .decode((self, sess))
            .collect::<Vec<_>>())
    }

    crate fn get_struct_field_names(
        &self,
        id: DefIndex,
        sess: &Session,
    ) -> Vec<Spanned<ast::Name>> {
        self.root.per_def.children.get(self, id).unwrap_or(Lazy::empty())
            .decode(self)
            .map(|index| respan(self.get_span(index, sess), self.item_name(index)))
            .collect()
    }

    // Translate a DefId from the current compilation environment to a DefId
    // for an external crate.
    fn reverse_translate_def_id(&self, did: DefId) -> Option<DefId> {
        for (local, &global) in self.cnum_map.iter_enumerated() {
            if global == did.krate {
                return Some(DefId {
                    krate: local,
                    index: did.index,
                });
            }
        }

        None
    }

    crate fn get_inherent_implementations_for_type(
        &self,
        tcx: TyCtxt<'tcx>,
        id: DefIndex,
    ) -> &'tcx [DefId] {
        tcx.arena.alloc_from_iter(
            self.root.per_def.inherent_impls.get(self, id).unwrap_or(Lazy::empty())
                .decode(self)
                .map(|index| self.local_def_id(index))
        )
    }

    crate fn get_implementations_for_trait(
        &self,
        tcx: TyCtxt<'tcx>,
        filter: Option<DefId>,
    ) -> &'tcx [DefId] {
        if self.is_proc_macro_crate() {
            // proc-macro crates export no trait impls.
            return &[]
        }

        // Do a reverse lookup beforehand to avoid touching the crate_num
        // hash map in the loop below.
        let filter = match filter.map(|def_id| self.reverse_translate_def_id(def_id)) {
            Some(Some(def_id)) => Some((def_id.krate.as_u32(), def_id.index)),
            Some(None) => return &[],
            None => None,
        };

        if let Some(filter) = filter {
            if let Some(impls) = self.trait_impls.get(&filter) {
                tcx.arena.alloc_from_iter(impls.decode(self).map(|idx| self.local_def_id(idx)))
            } else {
                &[]
            }
        } else {
            tcx.arena.alloc_from_iter(self.trait_impls.values().flat_map(|impls| {
                impls.decode(self).map(|idx| self.local_def_id(idx))
            }))
        }
    }

    crate fn get_trait_of_item(&self, id: DefIndex) -> Option<DefId> {
        let def_key = self.def_key(id);
        match def_key.disambiguated_data.data {
            DefPathData::TypeNs(..) | DefPathData::ValueNs(..) => (),
            // Not an associated item
            _ => return None,
        }
        def_key.parent.and_then(|parent_index| {
            match self.kind(parent_index) {
                EntryKind::Trait(_) |
                EntryKind::TraitAlias => Some(self.local_def_id(parent_index)),
                _ => None,
            }
        })
    }


    crate fn get_native_libraries(&self, sess: &Session) -> Vec<NativeLibrary> {
        if self.is_proc_macro_crate() {
            // Proc macro crates do not have any *target* native libraries.
            vec![]
        } else {
            self.root.native_libraries.decode((self, sess)).collect()
        }
    }

    crate fn get_foreign_modules(&self, tcx: TyCtxt<'tcx>) -> &'tcx [ForeignModule] {
        if self.is_proc_macro_crate() {
            // Proc macro crates do not have any *target* foreign modules.
            &[]
        } else {
            tcx.arena.alloc_from_iter(self.root.foreign_modules.decode((self, tcx.sess)))
        }
    }

    crate fn get_dylib_dependency_formats(
        &self,
        tcx: TyCtxt<'tcx>,
    ) -> &'tcx [(CrateNum, LinkagePreference)] {
        tcx.arena.alloc_from_iter(self.root
            .dylib_dependency_formats
            .decode(self)
            .enumerate()
            .flat_map(|(i, link)| {
                let cnum = CrateNum::new(i + 1);
                link.map(|link| (self.cnum_map[cnum], link))
            }))
    }

    crate fn get_missing_lang_items(&self, tcx: TyCtxt<'tcx>) -> &'tcx [lang_items::LangItem] {
        if self.is_proc_macro_crate() {
            // Proc macro crates do not depend on any target weak lang-items.
            &[]
        } else {
            tcx.arena.alloc_from_iter(self.root
                .lang_items_missing
                .decode(self))
        }
    }

    crate fn get_fn_param_names(&self, id: DefIndex) -> Vec<ast::Name> {
        let param_names = match self.kind(id) {
            EntryKind::Fn(data) |
            EntryKind::ForeignFn(data) => data.decode(self).param_names,
            EntryKind::Method(data) => data.decode(self).fn_data.param_names,
            _ => Lazy::empty(),
        };
        param_names.decode(self).collect()
    }

    crate fn exported_symbols(
        &self,
        tcx: TyCtxt<'tcx>,
    ) -> Vec<(ExportedSymbol<'tcx>, SymbolExportLevel)> {
        if self.is_proc_macro_crate() {
            // If this crate is a custom derive crate, then we're not even going to
            // link those in so we skip those crates.
            vec![]
        } else {
            self.root.exported_symbols.decode((self, tcx)).collect()
        }
    }

    crate fn get_rendered_const(&self, id: DefIndex) -> String {
        match self.kind(id) {
            EntryKind::Const(_, data) |
            EntryKind::AssocConst(_, _, data) => data.decode(self).0,
            _ => bug!(),
        }
    }

    crate fn get_macro(&self, id: DefIndex) -> MacroDef {
        match self.kind(id) {
            EntryKind::MacroDef(macro_def) => macro_def.decode(self),
            _ => bug!(),
        }
    }

    crate fn is_const_fn_raw(&self, id: DefIndex) -> bool {
        let constness = match self.kind(id) {
            EntryKind::Method(data) => data.decode(self).fn_data.constness,
            EntryKind::Fn(data) => data.decode(self).constness,
            EntryKind::Variant(..) | EntryKind::Struct(..) => hir::Constness::Const,
            _ => hir::Constness::NotConst,
        };
        constness == hir::Constness::Const
    }

    crate fn asyncness(&self, id: DefIndex) -> hir::IsAsync {
         match self.kind(id) {
            EntryKind::Fn(data) => data.decode(self).asyncness,
            EntryKind::Method(data) => data.decode(self).fn_data.asyncness,
            EntryKind::ForeignFn(data) => data.decode(self).asyncness,
            _ => bug!("asyncness: expected function kind"),
        }
    }

    crate fn is_foreign_item(&self, id: DefIndex) -> bool {
        match self.kind(id) {
            EntryKind::ForeignImmStatic |
            EntryKind::ForeignMutStatic |
            EntryKind::ForeignFn(_) => true,
            _ => false,
        }
    }

    crate fn static_mutability(&self, id: DefIndex) -> Option<hir::Mutability> {
        match self.kind(id) {
            EntryKind::ImmStatic |
            EntryKind::ForeignImmStatic => Some(hir::MutImmutable),
            EntryKind::MutStatic |
            EntryKind::ForeignMutStatic => Some(hir::MutMutable),
            _ => None,
        }
    }

    crate fn fn_sig(&self, id: DefIndex, tcx: TyCtxt<'tcx>) -> ty::PolyFnSig<'tcx> {
        self.root.per_def.fn_sig.get(self, id).unwrap().decode((self, tcx))
    }

    #[inline]
    crate fn def_key(&self, index: DefIndex) -> DefKey {
        let mut key = self.def_path_table.def_key(index);
        if self.is_proc_macro(index) {
            let name = self.raw_proc_macro(index).name();
            key.disambiguated_data.data = DefPathData::MacroNs(Symbol::intern(name));
        }
        key
    }

    // Returns the path leading to the thing with this `id`.
    crate fn def_path(&self, id: DefIndex) -> DefPath {
        debug!("def_path(cnum={:?}, id={:?})", self.cnum, id);
        DefPath::make(self.cnum, id, |parent| self.def_key(parent))
    }

    #[inline]
    crate fn def_path_hash(&self, index: DefIndex) -> DefPathHash {
        self.def_path_table.def_path_hash(index)
    }

    /// Imports the source_map from an external crate into the source_map of the crate
    /// currently being compiled (the "local crate").
    ///
    /// The import algorithm works analogous to how AST items are inlined from an
    /// external crate's metadata:
    /// For every SourceFile in the external source_map an 'inline' copy is created in the
    /// local source_map. The correspondence relation between external and local
    /// SourceFiles is recorded in the `ImportedSourceFile` objects returned from this
    /// function. When an item from an external crate is later inlined into this
    /// crate, this correspondence information is used to translate the span
    /// information of the inlined item so that it refers the correct positions in
    /// the local source_map (see `<decoder::DecodeContext as SpecializedDecoder<Span>>`).
    ///
    /// The import algorithm in the function below will reuse SourceFiles already
    /// existing in the local source_map. For example, even if the SourceFile of some
    /// source file of libstd gets imported many times, there will only ever be
    /// one SourceFile object for the corresponding file in the local source_map.
    ///
    /// Note that imported SourceFiles do not actually contain the source code of the
    /// file they represent, just information about length, line breaks, and
    /// multibyte characters. This information is enough to generate valid debuginfo
    /// for items inlined from other crates.
    ///
    /// Proc macro crates don't currently export spans, so this function does not have
    /// to work for them.
    pub(super) fn imported_source_files(
        &'a self,
        local_source_map: &source_map::SourceMap,
    ) -> &[ImportedSourceFile] {
        self.source_map_import_info.init_locking(|| {
            let external_source_map = self.root.source_map.decode(self);

            external_source_map.map(|source_file_to_import| {
                // We can't reuse an existing SourceFile, so allocate a new one
                // containing the information we need.
                let syntax_pos::SourceFile { name,
                                          name_was_remapped,
                                          src_hash,
                                          start_pos,
                                          end_pos,
                                          mut lines,
                                          mut multibyte_chars,
                                          mut non_narrow_chars,
                                          mut normalized_pos,
                                          name_hash,
                                          .. } = source_file_to_import;

                let source_length = (end_pos - start_pos).to_usize();

                // Translate line-start positions and multibyte character
                // position into frame of reference local to file.
                // `SourceMap::new_imported_source_file()` will then translate those
                // coordinates to their new global frame of reference when the
                // offset of the SourceFile is known.
                for pos in &mut lines {
                    *pos = *pos - start_pos;
                }
                for mbc in &mut multibyte_chars {
                    mbc.pos = mbc.pos - start_pos;
                }
                for swc in &mut non_narrow_chars {
                    *swc = *swc - start_pos;
                }
                for np in &mut normalized_pos {
                    np.pos = np.pos - start_pos;
                }

                let local_version = local_source_map.new_imported_source_file(name,
                                                                       name_was_remapped,
                                                                       self.cnum.as_u32(),
                                                                       src_hash,
                                                                       name_hash,
                                                                       source_length,
                                                                       lines,
                                                                       multibyte_chars,
                                                                       non_narrow_chars,
                                                                       normalized_pos);
                debug!("CrateMetaData::imported_source_files alloc \
                        source_file {:?} original (start_pos {:?} end_pos {:?}) \
                        translated (start_pos {:?} end_pos {:?})",
                       local_version.name, start_pos, end_pos,
                       local_version.start_pos, local_version.end_pos);

                ImportedSourceFile {
                    original_start_pos: start_pos,
                    original_end_pos: end_pos,
                    translated_source_file: local_version,
                }
            }).collect()
        })
    }

    /// Get the `DepNodeIndex` corresponding this crate. The result of this
    /// method is cached in the `dep_node_index` field.
    pub(super) fn get_crate_dep_node_index(&self, tcx: TyCtxt<'tcx>) -> DepNodeIndex {
        let mut dep_node_index = self.dep_node_index.load();

        if unlikely!(dep_node_index == DepNodeIndex::INVALID) {
            // We have not cached the DepNodeIndex for this upstream crate yet,
            // so use the dep-graph to find it out and cache it.
            // Note that multiple threads can enter this block concurrently.
            // That is fine because the DepNodeIndex remains constant
            // throughout the whole compilation session, and multiple stores
            // would always write the same value.

            let def_path_hash = self.def_path_hash(CRATE_DEF_INDEX);
            let dep_node = def_path_hash.to_dep_node(dep_graph::DepKind::CrateMetadata);

            dep_node_index = tcx.dep_graph.dep_node_index_of(&dep_node);
            assert!(dep_node_index != DepNodeIndex::INVALID);
            self.dep_node_index.store(dep_node_index);
        }

        dep_node_index
    }
}

// Cannot be implemented on 'ProcMacro', as libproc_macro
// does not depend on libsyntax
fn macro_kind(raw: &ProcMacro) -> MacroKind {
    match raw {
        ProcMacro::CustomDerive { .. } => MacroKind::Derive,
        ProcMacro::Attr { .. } => MacroKind::Attr,
        ProcMacro::Bang { .. } => MacroKind::Bang
    }
}

impl<'tcx> EntryKind<'tcx> {
    fn def_kind(&self) -> Option<DefKind> {
        Some(match *self {
            EntryKind::Const(..) => DefKind::Const,
            EntryKind::AssocConst(..) => DefKind::AssocConst,
            EntryKind::ImmStatic |
            EntryKind::MutStatic |
            EntryKind::ForeignImmStatic |
            EntryKind::ForeignMutStatic => DefKind::Static,
            EntryKind::Struct(_, _) => DefKind::Struct,
            EntryKind::Union(_, _) => DefKind::Union,
            EntryKind::Fn(_) |
            EntryKind::ForeignFn(_) => DefKind::Fn,
            EntryKind::Method(_) => DefKind::Method,
            EntryKind::Type => DefKind::TyAlias,
            EntryKind::TypeParam => DefKind::TyParam,
            EntryKind::ConstParam => DefKind::ConstParam,
            EntryKind::OpaqueTy => DefKind::OpaqueTy,
            EntryKind::AssocType(_) => DefKind::AssocTy,
            EntryKind::AssocOpaqueTy(_) => DefKind::AssocOpaqueTy,
            EntryKind::Mod(_) => DefKind::Mod,
            EntryKind::Variant(_) => DefKind::Variant,
            EntryKind::Trait(_) => DefKind::Trait,
            EntryKind::TraitAlias => DefKind::TraitAlias,
            EntryKind::Enum(..) => DefKind::Enum,
            EntryKind::MacroDef(_) => DefKind::Macro(MacroKind::Bang),
            EntryKind::ForeignType => DefKind::ForeignTy,

            EntryKind::ForeignMod |
            EntryKind::GlobalAsm |
            EntryKind::Impl(_) |
            EntryKind::Field |
            EntryKind::Generator(_) |
            EntryKind::Closure => return None,
        })
    }
}

#[derive(Clone)]
pub struct CStore {
    metas: IndexVec<CrateNum, Option<Lrc<CrateMetadata>>>,
}

pub enum LoadedMacro {
    MacroDef(ast::Item, Edition),
    ProcMacro(SyntaxExtension),
}

impl Default for CStore {
    fn default() -> Self {
        CStore {
            // We add an empty entry for LOCAL_CRATE (which maps to zero) in
            // order to make array indices in `metas` match with the
            // corresponding `CrateNum`. This first entry will always remain
            // `None`.
            metas: IndexVec::from_elem_n(None, 1),
        }
    }
}

impl CStore {
    crate fn alloc_new_crate_num(&mut self) -> CrateNum {
        self.metas.push(None);
        CrateNum::new(self.metas.len() - 1)
    }

    crate fn get_crate_data(&self, cnum: CrateNum) -> &CrateMetadata {
        self.metas[cnum].as_ref()
            .unwrap_or_else(|| panic!("Failed to get crate data for {:?}", cnum))
    }

    crate fn set_crate_data(&mut self, cnum: CrateNum, data: CrateMetadata) {
        assert!(self.metas[cnum].is_none(), "Overwriting crate metadata entry");
        self.metas[cnum] = Some(Lrc::new(data));
    }

    crate fn iter_crate_data<I>(&self, mut i: I)
        where I: FnMut(CrateNum, &CrateMetadata)
    {
        for (k, v) in self.metas.iter_enumerated() {
            if let &Some(ref v) = v {
                i(k, v);
            }
        }
    }

    crate fn crate_dependencies_in_rpo(&self, krate: CrateNum) -> Vec<CrateNum> {
        let mut ordering = Vec::new();
        self.push_dependencies_in_postorder(&mut ordering, krate);
        ordering.reverse();
        ordering
    }

    crate fn push_dependencies_in_postorder(&self, ordering: &mut Vec<CrateNum>, krate: CrateNum) {
        if ordering.contains(&krate) {
            return;
        }

        let data = self.get_crate_data(krate);
        for &dep in data.dependencies.borrow().iter() {
            if dep != krate {
                self.push_dependencies_in_postorder(ordering, dep);
            }
        }

        ordering.push(krate);
    }

    crate fn do_postorder_cnums_untracked(&self) -> Vec<CrateNum> {
        let mut ordering = Vec::new();
        for (num, v) in self.metas.iter_enumerated() {
            if let &Some(_) = v {
                self.push_dependencies_in_postorder(&mut ordering, num);
            }
        }
        return ordering
    }
}
