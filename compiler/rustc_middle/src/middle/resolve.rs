//! This module contains types that carry name resolution results from `rustc_resolve` to a
//! consumer in another crate (e.g. AST lowering, metadata, or a query).

use rustc_ast::node_id::NodeMap;
use rustc_ast::{self as ast, NodeId};
use rustc_attr_ir::StrippedCfgItem;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::steal::Steal;
use rustc_data_structures::unord::{UnordMap, UnordSet};
use rustc_errors::{ErrorGuaranteed, LintBuffer};
use rustc_hir::def::{DefKind, Namespace, PerNS, Res};
use rustc_hir::def_id::{CrateNum, DefId, LocalDefId, LocalDefIdMap, LocalModId, ModId};
use rustc_hir::definitions::PerParentDisambiguatorState;
use rustc_hir::{MissingLifetimeKind, TraitCandidate};
use rustc_macros::{StableHash, TyDecodable, TyEncodable};
use rustc_span::{ExpnId, Ident, Span, Symbol};
use smallvec::SmallVec;

use crate::middle::privacy::EffectiveVisibilities;
use crate::ty::Visibility;

/// The result of resolving a path before lowering to HIR,
/// with "module" segments resolved and associated item
/// segments deferred to type checking.
/// `base_res` is the resolution of the resolved part of the
/// path, `unresolved_segments` is the number of unresolved
/// segments.
///
/// ```text
/// module::Type::AssocX::AssocY::MethodOrAssocType
/// ^~~~~~~~~~~~  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// base_res      unresolved_segments = 3
///
/// <T as Trait>::AssocX::AssocY::MethodOrAssocType
///       ^~~~~~~~~~~~~~  ^~~~~~~~~~~~~~~~~~~~~~~~~
///       base_res        unresolved_segments = 2
/// ```
#[derive(Copy, Clone, Debug)]
pub struct PartialRes {
    base_res: Res<NodeId>,
    unresolved_segments: usize,
}

impl PartialRes {
    #[inline]
    pub fn new(base_res: Res<NodeId>) -> Self {
        PartialRes { base_res, unresolved_segments: 0 }
    }

    #[inline]
    pub fn with_unresolved_segments(base_res: Res<NodeId>, mut unresolved_segments: usize) -> Self {
        if base_res == Res::Err {
            unresolved_segments = 0
        }
        PartialRes { base_res, unresolved_segments }
    }

    #[inline]
    pub fn base_res(&self) -> Res<NodeId> {
        self.base_res
    }

    #[inline]
    pub fn unresolved_segments(&self) -> usize {
        self.unresolved_segments
    }

    #[inline]
    pub fn full_res(&self) -> Option<Res<NodeId>> {
        (self.unresolved_segments == 0).then_some(self.base_res)
    }

    #[inline]
    pub fn expect_full_res(&self) -> Res<NodeId> {
        self.full_res().expect("unexpected unresolved segments")
    }
}

/// Resolution for a lifetime appearing in a type.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LifetimeRes {
    /// Successfully linked the lifetime to a generic parameter.
    Param {
        /// Id of the generic parameter that introduced it.
        param: LocalDefId,
        /// Id of the introducing place. That can be:
        /// - an item's id, for the item's generic parameters;
        /// - a TraitRef's ref_id, identifying the `for<...>` binder;
        /// - a FnPtr type's id.
        ///
        /// This information is used for impl-trait lifetime captures, to know when to or not to
        /// capture any given lifetime.
        binder: NodeId,
    },
    /// Created a generic parameter for an anonymous lifetime.
    Fresh {
        /// Id of the generic parameter that introduced it.
        ///
        /// Creating the associated `LocalDefId` is the responsibility of lowering.
        param: NodeId,
        /// Kind of elided lifetime
        kind: MissingLifetimeKind,
    },
    /// This variant is used for anonymous lifetimes that we did not resolve during
    /// late resolution. Those lifetimes will be inferred by typechecking.
    Infer,
    /// `'static` lifetime.
    Static,
    /// Resolution failure.
    Error(ErrorGuaranteed),
    /// HACK: This is used to recover the NodeId of an elided lifetime.
    ElidedAnchor { start: NodeId, end: NodeId },
}

/// A simplified version of `ImportKind` from resolve.
/// `DefId`s here correspond to `use` and `extern crate` items themselves, not their targets.
#[derive(Clone, Copy, Debug, TyEncodable, TyDecodable, StableHash)]
pub enum Reexport {
    Single(DefId),
    Glob(DefId),
    ExternCrate(DefId),
    MacroUse,
    MacroExport,
}

impl Reexport {
    pub fn id(self) -> Option<DefId> {
        match self {
            Reexport::Single(id) | Reexport::Glob(id) | Reexport::ExternCrate(id) => Some(id),
            Reexport::MacroUse | Reexport::MacroExport => None,
        }
    }
}

/// This structure is supposed to keep enough data to re-create `Decl`s for other crates
/// during name resolution. Right now the bindings are not recreated entirely precisely so we may
/// need to add more data in the future to correctly support macros 2.0, for example.
/// Module child can be either a proper item or a reexport (including private imports).
/// In case of reexport all the fields describe the reexport item itself, not what it refers to.
#[derive(Debug, TyEncodable, TyDecodable, StableHash)]
pub struct ModChild {
    /// Name of the item.
    pub ident: Ident,
    /// Resolution result corresponding to the item.
    /// Local variables cannot be exported, so this `Res` doesn't need the ID parameter.
    pub res: Res<!>,
    /// Visibility of the item.
    pub vis: Visibility<ModId>,
    /// Reexport chain linking this module child to its original reexported item.
    /// Empty if the module child is a proper item.
    pub reexport_chain: SmallVec<[Reexport; 2]>,
}

/// Same as `ModChild`, however, it includes ambiguity error.
#[derive(Debug, TyEncodable, TyDecodable, StableHash)]
pub struct AmbigModChild {
    pub main: ModChild,
    pub second: ModChild,
}

#[derive(Debug, StableHash)]
pub struct ResolverGlobalCtxt {
    pub visibilities_for_hashing: Vec<(LocalDefId, Visibility)>,
    /// Item with a given `LocalDefId` was defined during macro expansion with ID `ExpnId`.
    pub expn_that_defined: UnordMap<LocalDefId, ExpnId>,
    pub effective_visibilities: EffectiveVisibilities,
    // FIXME: This table contains ADTs reachable from macro 2.0.
    // Currently, reachability of a definition from a macro is determined by nominal visibility
    // (see `compute_effective_visibilities`). This is incorrect and leads to the necessity
    // of traversing ADT fields in `rustc_privacy`. Remove this workaround once the
    // correct reachability logic is implemented for macros.
    pub macro_reachable_adts: FxIndexMap<LocalDefId, FxIndexSet<LocalDefId>>,
    pub extern_crate_map: UnordMap<LocalDefId, CrateNum>,
    pub maybe_unused_trait_imports: FxIndexSet<LocalDefId>,
    pub module_children: LocalDefIdMap<Vec<ModChild>>,
    pub ambig_module_children: LocalDefIdMap<Vec<AmbigModChild>>,
    pub glob_map: FxIndexMap<LocalDefId, FxIndexSet<Symbol>>,
    pub main_def: Option<MainDefinition>,
    pub trait_impls: FxIndexMap<DefId, Vec<LocalDefId>>,
    /// A list of proc macro LocalDefIds, written out in the order in which
    /// they are declared in the static array generated by proc_macro_harness.
    pub proc_macros: Vec<LocalDefId>,
    /// Mapping from ident span to path span for paths that don't exist as written, but that
    /// exist under `std`. For example, wrote `str::from_utf8` instead of `std::str::from_utf8`.
    pub confused_type_with_std_module: FxIndexMap<Span, Span>,
    pub doc_link_resolutions: FxIndexMap<LocalModId, DocLinkResMap>,
    pub doc_link_traits_in_scope: FxIndexMap<LocalModId, Vec<DefId>>,
    pub all_macro_rules: UnordSet<Symbol>,
    pub stripped_cfg_items: Vec<StrippedCfgItem>,
    // Information about delegations which is used when handling recursive delegations
    // and ensures easy access to delegation-only `LocalDefId`s.
    pub delegation_infos: FxIndexMap<LocalDefId, DelegationInfo>,
}

#[derive(Debug)]
pub struct PerOwnerResolverData<'tcx> {
    pub node_id_to_def_id: NodeMap<LocalDefId> = Default::default(),
    /// Whether lifetime elision was successful.
    pub lifetime_elision_allowed: bool = false,
    /// Resolutions for labels. Maps from NodeId of the break/continue expression to the NodeId of
    /// their corresponding blocks or loops.
    pub label_res_map: NodeMap<NodeId> = Default::default(),
    /// Resolutions for lifetimes.
    pub lifetimes_res_map: NodeMap<LifetimeRes> = Default::default(),

    pub trait_map: NodeMap<&'tcx [TraitCandidate<'tcx>]> = Default::default(),

    /// Resolution for import nodes, which have multiple resolutions in different namespaces.
    pub import_res: PerNS<Option<Res<NodeId>>> = Default::default(),
    /// Lifetime parameters that lowering will have to introduce.
    pub extra_lifetime_params_map: NodeMap<Vec<(Ident, NodeId, MissingLifetimeKind)>> =
        Default::default(),

    /// The id of the owner
    pub id: NodeId,
    /// The `DefId` of the owner, can't be found in `node_id_to_def_id`.
    pub def_id: LocalDefId,
}

impl<'tcx> PerOwnerResolverData<'tcx> {
    pub fn new(id: NodeId, def_id: LocalDefId) -> PerOwnerResolverData<'tcx> {
        PerOwnerResolverData { id, def_id, .. }
    }

    /// Obtains resolution for a label with the given `NodeId`.
    pub fn get_label_res(&self, id: NodeId) -> Option<NodeId> {
        self.label_res_map.get(&id).copied()
    }

    /// Obtains resolution for a lifetime with the given `NodeId`.
    pub fn get_lifetime_res(&self, id: NodeId) -> Option<LifetimeRes> {
        self.lifetimes_res_map.get(&id).copied()
    }

    /// Obtain the list of lifetimes parameters to add to an item.
    ///
    /// Extra lifetime parameters should only be added in places that can appear
    /// as a `binder` in `LifetimeRes`.
    ///
    /// The extra lifetimes that appear from the parenthesized `Fn`-trait desugaring
    /// should appear at the enclosing `PolyTraitRef`.
    pub fn extra_lifetime_params(&self, id: NodeId) -> &[(Ident, NodeId, MissingLifetimeKind)] {
        self.extra_lifetime_params_map.get(&id).map_or(&[], |v| &v[..])
    }
}

/// Resolutions that should only be used for lowering.
/// This struct is meant to be consumed by lowering.
#[derive(Debug)]
pub struct ResolverAstLowering<'tcx> {
    /// Resolutions for nodes that have a single resolution.
    pub partial_res_map: NodeMap<PartialRes>,

    pub next_node_id: NodeId,

    pub owners: NodeMap<PerOwnerResolverData<'tcx>>,

    /// Lints that were emitted by the resolver and early lints.
    pub lint_buffer: Steal<LintBuffer>,

    pub disambiguators: LocalDefIdMap<Steal<PerParentDisambiguatorState>>,
}

#[derive(Debug, StableHash)]
pub struct DelegationInfo {
    // `DefId` (either the resolution at delegation.id or item_id in case of a trait impl) for
    // signature resolution, for details see
    // https://github.com/rust-lang/rust/issues/118212#issuecomment-2160686914.
    /// Refers to the next element in a delegation resolution chain. Usually points to the final
    /// resolution, as most "chains" are just one step to a trait or an impl.
    pub resolution_id: Result<DefId, ErrorGuaranteed>,
}

#[derive(Clone, Copy, Debug, StableHash)]
pub struct MainDefinition {
    pub res: Res<NodeId>,
    pub is_import: bool,
    pub span: Span,
}

impl MainDefinition {
    pub fn opt_fn_def_id(self) -> Option<DefId> {
        if let Res::Def(DefKind::Fn, def_id) = self.res { Some(def_id) } else { None }
    }
}

// FxIndexMap is necessary because its data ends up in .rmeta files,
// so its iteration order must be consistent. See #159677 for context.
pub type DocLinkResMap = FxIndexMap<(Symbol, Namespace), Option<Res<NodeId>>>;

/// Fragment of the AST according to "HIR owner" semantics.
///
/// This is used to map each `LocalDefId` to its content's AST.
///
/// This type isn't produced by name resolution but it is paired with `ResolverAstLowering` so this
/// is as good a place as any for it.
#[derive(Debug)]
pub enum AstOwner {
    /// This definition does not correspond to a HIR owner.
    NonOwner,
    /// This definition corresponds to a nested `use` tree.
    /// The `LocalDefId` points to its HIR owner.
    NestedUseTree(LocalDefId),
    Crate(Box<ast::Crate>),
    Item(Box<ast::Item>),
    TraitItem(Box<ast::AssocItem>),
    ImplItem(Box<ast::AssocItem>),
    ForeignItem(Box<ast::ForeignItem>),
}
