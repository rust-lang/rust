// ignore-tidy-filelength
//! Name resolution for lifetimes.
//!
//! Name resolution for lifetimes follows *much* simpler rules than the
//! full resolve. For example, lifetime names are never exported or
//! used between functions, and they operate in a purely top-down
//! way. Therefore, we break lifetime name resolution into a separate pass.

use crate::late::diagnostics::{ForLifetimeSpanType, MissingLifetimeSpot};
use rustc_ast::walk_list;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap};
use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefIdMap, LocalDefId};
use rustc_hir::hir_id::ItemLocalId;
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::{GenericArg, GenericParam, LifetimeName, Node, ParamName, QPath};
use rustc_hir::{GenericParamKind, HirIdMap, HirIdSet, LifetimeParamKind};
use rustc_middle::hir::map::Map;
use rustc_middle::middle::resolve_lifetime::*;
use rustc_middle::ty::{self, DefIdTree, GenericParamDefKind, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_session::lint;
use rustc_span::def_id::DefId;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::Span;
use std::borrow::Cow;
use std::cell::Cell;
use std::fmt;
use std::mem::take;

use tracing::{debug, span, Level};

// This counts the no of times a lifetime is used
#[derive(Clone, Copy, Debug)]
pub enum LifetimeUseSet<'tcx> {
    One(&'tcx hir::Lifetime),
    Many,
}

trait RegionExt {
    fn early(hir_map: &Map<'_>, index: &mut u32, param: &GenericParam<'_>) -> (ParamName, Region);

    fn late(index: u32, hir_map: &Map<'_>, param: &GenericParam<'_>) -> (ParamName, Region);

    fn late_anon(named_late_bound_vars: u32, index: &Cell<u32>) -> Region;

    fn id(&self) -> Option<DefId>;

    fn shifted(self, amount: u32) -> Region;

    fn shifted_out_to_binder(self, binder: ty::DebruijnIndex) -> Region;

    fn subst<'a, L>(self, params: L, map: &NamedRegionMap) -> Option<Region>
    where
        L: Iterator<Item = &'a hir::Lifetime>;
}

impl RegionExt for Region {
    fn early(hir_map: &Map<'_>, index: &mut u32, param: &GenericParam<'_>) -> (ParamName, Region) {
        let i = *index;
        *index += 1;
        let def_id = hir_map.local_def_id(param.hir_id);
        let origin = LifetimeDefOrigin::from_param(param);
        debug!("Region::early: index={} def_id={:?}", i, def_id);
        (param.name.normalize_to_macros_2_0(), Region::EarlyBound(i, def_id.to_def_id(), origin))
    }

    fn late(idx: u32, hir_map: &Map<'_>, param: &GenericParam<'_>) -> (ParamName, Region) {
        let depth = ty::INNERMOST;
        let def_id = hir_map.local_def_id(param.hir_id);
        let origin = LifetimeDefOrigin::from_param(param);
        debug!(
            "Region::late: idx={:?}, param={:?} depth={:?} def_id={:?} origin={:?}",
            idx, param, depth, def_id, origin,
        );
        (
            param.name.normalize_to_macros_2_0(),
            Region::LateBound(depth, idx, def_id.to_def_id(), origin),
        )
    }

    fn late_anon(named_late_bound_vars: u32, index: &Cell<u32>) -> Region {
        let i = index.get();
        index.set(i + 1);
        let depth = ty::INNERMOST;
        Region::LateBoundAnon(depth, named_late_bound_vars + i, i)
    }

    fn id(&self) -> Option<DefId> {
        match *self {
            Region::Static | Region::LateBoundAnon(..) => None,

            Region::EarlyBound(_, id, _) | Region::LateBound(_, _, id, _) | Region::Free(_, id) => {
                Some(id)
            }
        }
    }

    fn shifted(self, amount: u32) -> Region {
        match self {
            Region::LateBound(debruijn, idx, id, origin) => {
                Region::LateBound(debruijn.shifted_in(amount), idx, id, origin)
            }
            Region::LateBoundAnon(debruijn, index, anon_index) => {
                Region::LateBoundAnon(debruijn.shifted_in(amount), index, anon_index)
            }
            _ => self,
        }
    }

    fn shifted_out_to_binder(self, binder: ty::DebruijnIndex) -> Region {
        match self {
            Region::LateBound(debruijn, index, id, origin) => {
                Region::LateBound(debruijn.shifted_out_to_binder(binder), index, id, origin)
            }
            Region::LateBoundAnon(debruijn, index, anon_index) => {
                Region::LateBoundAnon(debruijn.shifted_out_to_binder(binder), index, anon_index)
            }
            _ => self,
        }
    }

    fn subst<'a, L>(self, mut params: L, map: &NamedRegionMap) -> Option<Region>
    where
        L: Iterator<Item = &'a hir::Lifetime>,
    {
        if let Region::EarlyBound(index, _, _) = self {
            params.nth(index as usize).and_then(|lifetime| map.defs.get(&lifetime.hir_id).cloned())
        } else {
            Some(self)
        }
    }
}

/// Maps the id of each lifetime reference to the lifetime decl
/// that it corresponds to.
///
/// FIXME. This struct gets converted to a `ResolveLifetimes` for
/// actual use. It has the same data, but indexed by `LocalDefId`.  This
/// is silly.
#[derive(Debug, Default)]
struct NamedRegionMap {
    // maps from every use of a named (not anonymous) lifetime to a
    // `Region` describing how that region is bound
    defs: HirIdMap<Region>,

    // the set of lifetime def ids that are late-bound; a region can
    // be late-bound if (a) it does NOT appear in a where-clause and
    // (b) it DOES appear in the arguments.
    late_bound: HirIdSet,

    // Maps relevant hir items to the bound vars on them. These include:
    // - function defs
    // - function pointers
    // - closures
    // - trait refs
    // - bound types (like `T` in `for<'a> T<'a>: Foo`)
    late_bound_vars: HirIdMap<Vec<ty::BoundVariableKind>>,

    // maps `PathSegment` `HirId`s to lifetime scopes.
    scope_for_path: Option<FxHashMap<LocalDefId, FxHashMap<ItemLocalId, LifetimeScopeForPath>>>,
}

crate struct LifetimeContext<'a, 'tcx> {
    crate tcx: TyCtxt<'tcx>,
    map: &'a mut NamedRegionMap,
    scope: ScopeRef<'a>,

    /// Used to disallow the use of in-band lifetimes in `fn` or `Fn` syntax.
    is_in_fn_syntax: bool,

    is_in_const_generic: bool,

    /// Indicates that we only care about the definition of a trait. This should
    /// be false if the `Item` we are resolving lifetimes for is not a trait or
    /// we eventually need lifetimes resolve for trait items.
    trait_definition_only: bool,

    /// List of labels in the function/method currently under analysis.
    labels_in_fn: Vec<Ident>,

    /// Cache for cross-crate per-definition object lifetime defaults.
    xcrate_object_lifetime_defaults: DefIdMap<Vec<ObjectLifetimeDefault>>,

    lifetime_uses: &'a mut DefIdMap<LifetimeUseSet<'tcx>>,

    /// When encountering an undefined named lifetime, we will suggest introducing it in these
    /// places.
    crate missing_named_lifetime_spots: Vec<MissingLifetimeSpot<'tcx>>,
}

#[derive(Debug)]
enum Scope<'a> {
    /// Declares lifetimes, and each can be early-bound or late-bound.
    /// The `DebruijnIndex` of late-bound lifetimes starts at `1` and
    /// it should be shifted by the number of `Binder`s in between the
    /// declaration `Binder` and the location it's referenced from.
    Binder {
        /// We use an IndexMap here because we want these lifetimes in order
        /// for diagnostics.
        lifetimes: FxIndexMap<hir::ParamName, Region>,

        /// if we extend this scope with another scope, what is the next index
        /// we should use for an early-bound region?
        next_early_index: u32,

        /// Flag is set to true if, in this binder, `'_` would be
        /// equivalent to a "single-use region". This is true on
        /// impls, but not other kinds of items.
        track_lifetime_uses: bool,

        /// Whether or not this binder would serve as the parent
        /// binder for opaque types introduced within. For example:
        ///
        /// ```text
        ///     fn foo<'a>() -> impl for<'b> Trait<Item = impl Trait2<'a>>
        /// ```
        ///
        /// Here, the opaque types we create for the `impl Trait`
        /// and `impl Trait2` references will both have the `foo` item
        /// as their parent. When we get to `impl Trait2`, we find
        /// that it is nested within the `for<>` binder -- this flag
        /// allows us to skip that when looking for the parent binder
        /// of the resulting opaque type.
        opaque_type_parent: bool,

        scope_type: BinderScopeType,

        /// The late bound vars for a given item are stored by `HirId` to be
        /// queried later. However, if we enter an elision scope, we have to
        /// later append the elided bound vars to the list and need to know what
        /// to append to.
        hir_id: hir::HirId,

        s: ScopeRef<'a>,
    },

    /// Lifetimes introduced by a fn are scoped to the call-site for that fn,
    /// if this is a fn body, otherwise the original definitions are used.
    /// Unspecified lifetimes are inferred, unless an elision scope is nested,
    /// e.g., `(&T, fn(&T) -> &T);` becomes `(&'_ T, for<'a> fn(&'a T) -> &'a T)`.
    Body {
        id: hir::BodyId,
        s: ScopeRef<'a>,
    },

    /// A scope which either determines unspecified lifetimes or errors
    /// on them (e.g., due to ambiguity). For more details, see `Elide`.
    Elision {
        elide: Elide,
        s: ScopeRef<'a>,
    },

    /// Use a specific lifetime (if `Some`) or leave it unset (to be
    /// inferred in a function body or potentially error outside one),
    /// for the default choice of lifetime in a trait object type.
    ObjectLifetimeDefault {
        lifetime: Option<Region>,
        s: ScopeRef<'a>,
    },

    /// When we have nested trait refs, we concanetate late bound vars for inner
    /// trait refs from outer ones. But we also need to include any HRTB
    /// lifetimes encountered when identifying the trait that an associated type
    /// is declared on.
    Supertrait {
        lifetimes: Vec<ty::BoundVariableKind>,
        s: ScopeRef<'a>,
    },

    TraitRefBoundary {
        s: ScopeRef<'a>,
    },

    Root,
}

#[derive(Copy, Clone, Debug)]
enum BinderScopeType {
    /// Any non-concatenating binder scopes.
    Normal,
    /// Within a syntactic trait ref, there may be multiple poly trait refs that
    /// are nested (under the `associcated_type_bounds` feature). The binders of
    /// the innner poly trait refs are extended from the outer poly trait refs
    /// and don't increase the late bound depth. If you had
    /// `T: for<'a>  Foo<Bar: for<'b> Baz<'a, 'b>>`, then the `for<'b>` scope
    /// would be `Concatenating`. This also used in trait refs in where clauses
    /// where we have two binders `for<> T: for<> Foo` (I've intentionally left
    /// out any lifetimes because they aren't needed to show the two scopes).
    /// The inner `for<>` has a scope of `Concatenating`.
    Concatenating,
}

// A helper struct for debugging scopes without printing parent scopes
struct TruncatedScopeDebug<'a>(&'a Scope<'a>);

impl<'a> fmt::Debug for TruncatedScopeDebug<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Scope::Binder {
                lifetimes,
                next_early_index,
                track_lifetime_uses,
                opaque_type_parent,
                scope_type,
                hir_id,
                s: _,
            } => f
                .debug_struct("Binder")
                .field("lifetimes", lifetimes)
                .field("next_early_index", next_early_index)
                .field("track_lifetime_uses", track_lifetime_uses)
                .field("opaque_type_parent", opaque_type_parent)
                .field("scope_type", scope_type)
                .field("hir_id", hir_id)
                .field("s", &"..")
                .finish(),
            Scope::Body { id, s: _ } => {
                f.debug_struct("Body").field("id", id).field("s", &"..").finish()
            }
            Scope::Elision { elide, s: _ } => {
                f.debug_struct("Elision").field("elide", elide).field("s", &"..").finish()
            }
            Scope::ObjectLifetimeDefault { lifetime, s: _ } => f
                .debug_struct("ObjectLifetimeDefault")
                .field("lifetime", lifetime)
                .field("s", &"..")
                .finish(),
            Scope::Supertrait { lifetimes, s: _ } => f
                .debug_struct("Supertrait")
                .field("lifetimes", lifetimes)
                .field("s", &"..")
                .finish(),
            Scope::TraitRefBoundary { s: _ } => f.debug_struct("TraitRefBoundary").finish(),
            Scope::Root => f.debug_struct("Root").finish(),
        }
    }
}

#[derive(Clone, Debug)]
enum Elide {
    /// Use a fresh anonymous late-bound lifetime each time, by
    /// incrementing the counter to generate sequential indices. All
    /// anonymous lifetimes must start *after* named bound vars.
    FreshLateAnon(u32, Cell<u32>),
    /// Always use this one lifetime.
    Exact(Region),
    /// Less or more than one lifetime were found, error on unspecified.
    Error(Vec<ElisionFailureInfo>),
    /// Forbid lifetime elision inside of a larger scope where it would be
    /// permitted. For example, in let position impl trait.
    Forbid,
}

#[derive(Clone, Debug)]
crate struct ElisionFailureInfo {
    /// Where we can find the argument pattern.
    parent: Option<hir::BodyId>,
    /// The index of the argument in the original definition.
    index: usize,
    lifetime_count: usize,
    have_bound_regions: bool,
    crate span: Span,
}

type ScopeRef<'a> = &'a Scope<'a>;

const ROOT_SCOPE: ScopeRef<'static> = &Scope::Root;

pub fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers {
        resolve_lifetimes_trait_definition,
        resolve_lifetimes,

        named_region_map: |tcx, id| resolve_lifetimes_for(tcx, id).defs.get(&id),
        is_late_bound_map,
        object_lifetime_defaults_map: |tcx, id| {
            let hir_id = tcx.hir().local_def_id_to_hir_id(id);
            match tcx.hir().find(hir_id) {
                Some(Node::Item(item)) => compute_object_lifetime_defaults(tcx, item),
                _ => None,
            }
        },
        late_bound_vars_map: |tcx, id| resolve_lifetimes_for(tcx, id).late_bound_vars.get(&id),
        lifetime_scope_map: |tcx, id| {
            let item_id = item_for(tcx, id);
            do_resolve(tcx, item_id, false, true).scope_for_path.unwrap().remove(&id)
        },

        ..*providers
    };
}

/// Like `resolve_lifetimes`, but does not resolve lifetimes for trait items.
/// Also does not generate any diagnostics.
///
/// This is ultimately a subset of the `resolve_lifetimes` work. It effectively
/// resolves lifetimes only within the trait "header" -- that is, the trait
/// and supertrait list. In contrast, `resolve_lifetimes` resolves all the
/// lifetimes within the trait and its items. There is room to refactor this,
/// for example to resolve lifetimes for each trait item in separate queries,
/// but it's convenient to do the entire trait at once because the lifetimes
/// from the trait definition are in scope within the trait items as well.
///
/// The reason for this separate call is to resolve what would otherwise
/// be a cycle. Consider this example:
///
/// ```rust
/// trait Base<'a> {
///     type BaseItem;
/// }
/// trait Sub<'b>: for<'a> Base<'a> {
///    type SubItem: Sub<BaseItem = &'b u32>;
/// }
/// ```
///
/// When we resolve `Sub` and all its items, we also have to resolve `Sub<BaseItem = &'b u32>`.
/// To figure out the index of `'b`, we have to know about the supertraits
/// of `Sub` so that we can determine that the `for<'a>` will be in scope.
/// (This is because we -- currently at least -- flatten all the late-bound
/// lifetimes into a single binder.) This requires us to resolve the
/// *trait definition* of `Sub`; basically just enough lifetime information
/// to look at the supertraits.
#[tracing::instrument(level = "debug", skip(tcx))]
fn resolve_lifetimes_trait_definition(
    tcx: TyCtxt<'_>,
    local_def_id: LocalDefId,
) -> ResolveLifetimes {
    convert_named_region_map(do_resolve(tcx, local_def_id, true, false))
}

/// Computes the `ResolveLifetimes` map that contains data for an entire `Item`.
/// You should not read the result of this query directly, but rather use
/// `named_region_map`, `is_late_bound_map`, etc.
#[tracing::instrument(level = "debug", skip(tcx))]
fn resolve_lifetimes(tcx: TyCtxt<'_>, local_def_id: LocalDefId) -> ResolveLifetimes {
    convert_named_region_map(do_resolve(tcx, local_def_id, false, false))
}

fn do_resolve(
    tcx: TyCtxt<'_>,
    local_def_id: LocalDefId,
    trait_definition_only: bool,
    with_scope_for_path: bool,
) -> NamedRegionMap {
    let item = tcx.hir().expect_item(tcx.hir().local_def_id_to_hir_id(local_def_id));
    let mut named_region_map = NamedRegionMap {
        defs: Default::default(),
        late_bound: Default::default(),
        late_bound_vars: Default::default(),
        scope_for_path: with_scope_for_path.then(|| Default::default()),
    };
    let mut visitor = LifetimeContext {
        tcx,
        map: &mut named_region_map,
        scope: ROOT_SCOPE,
        is_in_fn_syntax: false,
        is_in_const_generic: false,
        trait_definition_only,
        labels_in_fn: vec![],
        xcrate_object_lifetime_defaults: Default::default(),
        lifetime_uses: &mut Default::default(),
        missing_named_lifetime_spots: vec![],
    };
    visitor.visit_item(item);

    named_region_map
}

fn convert_named_region_map(named_region_map: NamedRegionMap) -> ResolveLifetimes {
    let mut rl = ResolveLifetimes::default();

    for (hir_id, v) in named_region_map.defs {
        let map = rl.defs.entry(hir_id.owner).or_default();
        map.insert(hir_id.local_id, v);
    }
    for hir_id in named_region_map.late_bound {
        let map = rl.late_bound.entry(hir_id.owner).or_default();
        map.insert(hir_id.local_id);
    }
    for (hir_id, v) in named_region_map.late_bound_vars {
        let map = rl.late_bound_vars.entry(hir_id.owner).or_default();
        map.insert(hir_id.local_id, v);
    }

    debug!(?rl.defs);
    rl
}

/// Given `any` owner (structs, traits, trait methods, etc.), does lifetime resolution.
/// There are two important things this does.
/// First, we have to resolve lifetimes for
/// the entire *`Item`* that contains this owner, because that's the largest "scope"
/// where we can have relevant lifetimes.
/// Second, if we are asking for lifetimes in a trait *definition*, we use `resolve_lifetimes_trait_definition`
/// instead of `resolve_lifetimes`, which does not descend into the trait items and does not emit diagnostics.
/// This allows us to avoid cycles. Importantly, if we ask for lifetimes for lifetimes that have an owner
/// other than the trait itself (like the trait methods or associated types), then we just use the regular
/// `resolve_lifetimes`.
fn resolve_lifetimes_for<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> &'tcx ResolveLifetimes {
    let item_id = item_for(tcx, def_id);
    if item_id == def_id {
        let item = tcx.hir().item(hir::ItemId { def_id: item_id });
        match item.kind {
            hir::ItemKind::Trait(..) => tcx.resolve_lifetimes_trait_definition(item_id),
            _ => tcx.resolve_lifetimes(item_id),
        }
    } else {
        tcx.resolve_lifetimes(item_id)
    }
}

/// Finds the `Item` that contains the given `LocalDefId`
fn item_for(tcx: TyCtxt<'_>, local_def_id: LocalDefId) -> LocalDefId {
    let hir_id = tcx.hir().local_def_id_to_hir_id(local_def_id);
    match tcx.hir().find(hir_id) {
        Some(Node::Item(item)) => {
            return item.def_id;
        }
        _ => {}
    }
    let item = {
        let hir = tcx.hir();
        let mut parent_iter = hir.parent_iter(hir_id);
        loop {
            let node = parent_iter.next().map(|n| n.1);
            match node {
                Some(hir::Node::Item(item)) => break item.def_id,
                Some(hir::Node::Crate(_)) | None => bug!("Called `item_for` on an Item."),
                _ => {}
            }
        }
    };
    item
}

fn is_late_bound_map<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> Option<(LocalDefId, &'tcx FxHashSet<ItemLocalId>)> {
    match tcx.def_kind(def_id) {
        DefKind::AnonConst => {
            let mut def_id = tcx
                .parent(def_id.to_def_id())
                .unwrap_or_else(|| bug!("anon const or closure without a parent"));
            // We search for the next outer anon const or fn here
            // while skipping closures.
            //
            // Note that for `AnonConst` we still just recurse until we
            // find a function body, but who cares :shrug:
            while tcx.is_closure(def_id) {
                def_id = tcx
                    .parent(def_id)
                    .unwrap_or_else(|| bug!("anon const or closure without a parent"));
            }

            tcx.is_late_bound_map(def_id.expect_local())
        }
        _ => resolve_lifetimes_for(tcx, def_id).late_bound.get(&def_id).map(|lt| (def_id, lt)),
    }
}

/// In traits, there is an implicit `Self` type parameter which comes before the generics.
/// We have to account for this when computing the index of the other generic parameters.
/// This function returns whether there is such an implicit parameter defined on the given item.
fn sub_items_have_self_param(node: &hir::ItemKind<'_>) -> bool {
    matches!(*node, hir::ItemKind::Trait(..) | hir::ItemKind::TraitAlias(..))
}

fn late_region_as_bound_region<'tcx>(tcx: TyCtxt<'tcx>, region: &Region) -> ty::BoundVariableKind {
    match region {
        Region::LateBound(_, _, def_id, _) => {
            let name = tcx.hir().name(tcx.hir().local_def_id_to_hir_id(def_id.expect_local()));
            ty::BoundVariableKind::Region(ty::BrNamed(*def_id, name))
        }
        Region::LateBoundAnon(_, _, anon_idx) => {
            ty::BoundVariableKind::Region(ty::BrAnon(*anon_idx))
        }
        _ => bug!("{:?} is not a late region", region),
    }
}

#[tracing::instrument(level = "debug")]
fn get_lifetime_scopes_for_path(mut scope: &Scope<'_>) -> LifetimeScopeForPath {
    let mut available_lifetimes = vec![];
    loop {
        match scope {
            Scope::Binder { lifetimes, s, .. } => {
                available_lifetimes.extend(lifetimes.keys().filter_map(|p| match p {
                    hir::ParamName::Plain(ident) => Some(ident.name.to_string()),
                    _ => None,
                }));
                scope = s;
            }
            Scope::Body { s, .. } => {
                scope = s;
            }
            Scope::Elision { elide, s } => {
                if let Elide::Exact(_) = elide {
                    return LifetimeScopeForPath::Elided;
                } else {
                    scope = s;
                }
            }
            Scope::ObjectLifetimeDefault { s, .. } => {
                scope = s;
            }
            Scope::Root => {
                return LifetimeScopeForPath::NonElided(available_lifetimes);
            }
            Scope::Supertrait { s, .. } | Scope::TraitRefBoundary { s, .. } => {
                scope = s;
            }
        }
    }
}

impl<'a, 'tcx> LifetimeContext<'a, 'tcx> {
    /// Returns the binders in scope and the type of `Binder` that should be created for a poly trait ref.
    fn poly_trait_ref_binder_info(&mut self) -> (Vec<ty::BoundVariableKind>, BinderScopeType) {
        let mut scope = self.scope;
        let mut supertrait_lifetimes = vec![];
        loop {
            match scope {
                Scope::Body { .. } | Scope::Root => {
                    break (vec![], BinderScopeType::Normal);
                }

                Scope::Elision { s, .. } | Scope::ObjectLifetimeDefault { s, .. } => {
                    scope = s;
                }

                Scope::Supertrait { s, lifetimes } => {
                    supertrait_lifetimes = lifetimes.clone();
                    scope = s;
                }

                Scope::TraitRefBoundary { .. } => {
                    // We should only see super trait lifetimes if there is a `Binder` above
                    assert!(supertrait_lifetimes.is_empty());
                    break (vec![], BinderScopeType::Normal);
                }

                Scope::Binder { hir_id, .. } => {
                    // Nested poly trait refs have the binders concatenated
                    let mut full_binders =
                        self.map.late_bound_vars.entry(*hir_id).or_default().clone();
                    full_binders.extend(supertrait_lifetimes.into_iter());
                    break (full_binders, BinderScopeType::Concatenating);
                }
            }
        }
    }
}
impl<'a, 'tcx> Visitor<'tcx> for LifetimeContext<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::All(self.tcx.hir())
    }

    // We want to nest trait/impl items in their parent, but nothing else.
    fn visit_nested_item(&mut self, _: hir::ItemId) {}

    fn visit_trait_item_ref(&mut self, ii: &'tcx hir::TraitItemRef) {
        if !self.trait_definition_only {
            intravisit::walk_trait_item_ref(self, ii)
        }
    }

    fn visit_nested_body(&mut self, body: hir::BodyId) {
        // Each body has their own set of labels, save labels.
        let saved = take(&mut self.labels_in_fn);
        let body = self.tcx.hir().body(body);
        extract_labels(self, body);
        self.with(Scope::Body { id: body.id(), s: self.scope }, |_, this| {
            this.visit_body(body);
        });
        self.labels_in_fn = saved;
    }

    fn visit_fn(
        &mut self,
        fk: intravisit::FnKind<'tcx>,
        fd: &'tcx hir::FnDecl<'tcx>,
        b: hir::BodyId,
        s: rustc_span::Span,
        hir_id: hir::HirId,
    ) {
        let name = match fk {
            intravisit::FnKind::ItemFn(id, _, _, _) => id.as_str(),
            intravisit::FnKind::Method(id, _, _) => id.as_str(),
            intravisit::FnKind::Closure => Symbol::intern("closure").as_str(),
        };
        let name: &str = &name;
        let span = span!(Level::DEBUG, "visit_fn", name);
        let _enter = span.enter();
        match fk {
            // Any `Binders` are handled elsewhere
            intravisit::FnKind::ItemFn(..) | intravisit::FnKind::Method(..) => {
                intravisit::walk_fn(self, fk, fd, b, s, hir_id)
            }
            intravisit::FnKind::Closure => {
                self.map.late_bound_vars.insert(hir_id, vec![]);
                let scope = Scope::Binder {
                    hir_id,
                    lifetimes: FxIndexMap::default(),
                    next_early_index: self.next_early_index(),
                    s: self.scope,
                    track_lifetime_uses: true,
                    opaque_type_parent: false,
                    scope_type: BinderScopeType::Normal,
                };
                self.with(scope, move |_old_scope, this| {
                    intravisit::walk_fn(this, fk, fd, b, s, hir_id)
                });
            }
        }
    }

    fn visit_item(&mut self, item: &'tcx hir::Item<'tcx>) {
        match &item.kind {
            hir::ItemKind::Impl(hir::Impl { of_trait, .. }) => {
                if let Some(of_trait) = of_trait {
                    self.map.late_bound_vars.insert(of_trait.hir_ref_id, Vec::default());
                }
            }
            _ => {}
        }
        match item.kind {
            hir::ItemKind::Fn(ref sig, ref generics, _) => {
                self.missing_named_lifetime_spots.push(generics.into());
                self.visit_early_late(None, item.hir_id(), &sig.decl, generics, |this| {
                    intravisit::walk_item(this, item);
                });
                self.missing_named_lifetime_spots.pop();
            }

            hir::ItemKind::ExternCrate(_)
            | hir::ItemKind::Use(..)
            | hir::ItemKind::Macro(..)
            | hir::ItemKind::Mod(..)
            | hir::ItemKind::ForeignMod { .. }
            | hir::ItemKind::GlobalAsm(..) => {
                // These sorts of items have no lifetime parameters at all.
                intravisit::walk_item(self, item);
            }
            hir::ItemKind::Static(..) | hir::ItemKind::Const(..) => {
                // No lifetime parameters, but implied 'static.
                let scope = Scope::Elision { elide: Elide::Exact(Region::Static), s: ROOT_SCOPE };
                self.with(scope, |_, this| intravisit::walk_item(this, item));
            }
            hir::ItemKind::OpaqueTy(hir::OpaqueTy { .. }) => {
                // Opaque types are visited when we visit the
                // `TyKind::OpaqueDef`, so that they have the lifetimes from
                // their parent opaque_ty in scope.
                //
                // The core idea here is that since OpaqueTys are generated with the impl Trait as
                // their owner, we can keep going until we find the Item that owns that. We then
                // conservatively add all resolved lifetimes. Otherwise we run into problems in
                // cases like `type Foo<'a> = impl Bar<As = impl Baz + 'a>`.
                for (_hir_id, node) in
                    self.tcx.hir().parent_iter(self.tcx.hir().local_def_id_to_hir_id(item.def_id))
                {
                    match node {
                        hir::Node::Item(parent_item) => {
                            let resolved_lifetimes: &ResolveLifetimes =
                                self.tcx.resolve_lifetimes(item_for(self.tcx, parent_item.def_id));
                            // We need to add *all* deps, since opaque tys may want them from *us*
                            for (&owner, defs) in resolved_lifetimes.defs.iter() {
                                defs.iter().for_each(|(&local_id, region)| {
                                    self.map.defs.insert(hir::HirId { owner, local_id }, *region);
                                });
                            }
                            for (&owner, late_bound) in resolved_lifetimes.late_bound.iter() {
                                late_bound.iter().for_each(|&local_id| {
                                    self.map.late_bound.insert(hir::HirId { owner, local_id });
                                });
                            }
                            for (&owner, late_bound_vars) in
                                resolved_lifetimes.late_bound_vars.iter()
                            {
                                late_bound_vars.iter().for_each(|(&local_id, late_bound_vars)| {
                                    self.map.late_bound_vars.insert(
                                        hir::HirId { owner, local_id },
                                        late_bound_vars.clone(),
                                    );
                                });
                            }
                            break;
                        }
                        hir::Node::Crate(_) => bug!("No Item about an OpaqueTy"),
                        _ => {}
                    }
                }
            }
            hir::ItemKind::TyAlias(_, ref generics)
            | hir::ItemKind::Enum(_, ref generics)
            | hir::ItemKind::Struct(_, ref generics)
            | hir::ItemKind::Union(_, ref generics)
            | hir::ItemKind::Trait(_, _, ref generics, ..)
            | hir::ItemKind::TraitAlias(ref generics, ..)
            | hir::ItemKind::Impl(hir::Impl { ref generics, .. }) => {
                self.missing_named_lifetime_spots.push(generics.into());

                // Impls permit `'_` to be used and it is equivalent to "some fresh lifetime name".
                // This is not true for other kinds of items.
                let track_lifetime_uses = matches!(item.kind, hir::ItemKind::Impl { .. });
                // These kinds of items have only early-bound lifetime parameters.
                let mut index = if sub_items_have_self_param(&item.kind) {
                    1 // Self comes before lifetimes
                } else {
                    0
                };
                let mut non_lifetime_count = 0;
                let lifetimes = generics
                    .params
                    .iter()
                    .filter_map(|param| match param.kind {
                        GenericParamKind::Lifetime { .. } => {
                            Some(Region::early(&self.tcx.hir(), &mut index, param))
                        }
                        GenericParamKind::Type { .. } | GenericParamKind::Const { .. } => {
                            non_lifetime_count += 1;
                            None
                        }
                    })
                    .collect();
                self.map.late_bound_vars.insert(item.hir_id(), vec![]);
                let scope = Scope::Binder {
                    hir_id: item.hir_id(),
                    lifetimes,
                    next_early_index: index + non_lifetime_count,
                    opaque_type_parent: true,
                    track_lifetime_uses,
                    scope_type: BinderScopeType::Normal,
                    s: ROOT_SCOPE,
                };
                self.with(scope, |old_scope, this| {
                    this.check_lifetime_params(old_scope, &generics.params);
                    let scope = Scope::TraitRefBoundary { s: this.scope };
                    this.with(scope, |_, this| {
                        intravisit::walk_item(this, item);
                    });
                });
                self.missing_named_lifetime_spots.pop();
            }
        }
    }

    fn visit_foreign_item(&mut self, item: &'tcx hir::ForeignItem<'tcx>) {
        match item.kind {
            hir::ForeignItemKind::Fn(ref decl, _, ref generics) => {
                self.visit_early_late(None, item.hir_id(), decl, generics, |this| {
                    intravisit::walk_foreign_item(this, item);
                })
            }
            hir::ForeignItemKind::Static(..) => {
                intravisit::walk_foreign_item(self, item);
            }
            hir::ForeignItemKind::Type => {
                intravisit::walk_foreign_item(self, item);
            }
        }
    }

    #[tracing::instrument(level = "debug", skip(self))]
    fn visit_ty(&mut self, ty: &'tcx hir::Ty<'tcx>) {
        match ty.kind {
            hir::TyKind::BareFn(ref c) => {
                let next_early_index = self.next_early_index();
                let was_in_fn_syntax = self.is_in_fn_syntax;
                self.is_in_fn_syntax = true;
                let lifetime_span: Option<Span> =
                    c.generic_params.iter().rev().find_map(|param| match param.kind {
                        GenericParamKind::Lifetime { .. } => Some(param.span),
                        _ => None,
                    });
                let (span, span_type) = if let Some(span) = lifetime_span {
                    (span.shrink_to_hi(), ForLifetimeSpanType::TypeTail)
                } else {
                    (ty.span.shrink_to_lo(), ForLifetimeSpanType::TypeEmpty)
                };
                self.missing_named_lifetime_spots
                    .push(MissingLifetimeSpot::HigherRanked { span, span_type });
                let (lifetimes, binders): (FxIndexMap<hir::ParamName, Region>, Vec<_>) = c
                    .generic_params
                    .iter()
                    .filter_map(|param| match param.kind {
                        GenericParamKind::Lifetime { .. } => Some(param),
                        _ => None,
                    })
                    .enumerate()
                    .map(|(late_bound_idx, param)| {
                        let pair = Region::late(late_bound_idx as u32, &self.tcx.hir(), param);
                        let r = late_region_as_bound_region(self.tcx, &pair.1);
                        (pair, r)
                    })
                    .unzip();
                self.map.late_bound_vars.insert(ty.hir_id, binders);
                let scope = Scope::Binder {
                    hir_id: ty.hir_id,
                    lifetimes,
                    s: self.scope,
                    next_early_index,
                    track_lifetime_uses: true,
                    opaque_type_parent: false,
                    scope_type: BinderScopeType::Normal,
                };
                self.with(scope, |old_scope, this| {
                    // a bare fn has no bounds, so everything
                    // contained within is scoped within its binder.
                    this.check_lifetime_params(old_scope, &c.generic_params);
                    intravisit::walk_ty(this, ty);
                });
                self.missing_named_lifetime_spots.pop();
                self.is_in_fn_syntax = was_in_fn_syntax;
            }
            hir::TyKind::TraitObject(bounds, ref lifetime, _) => {
                debug!(?bounds, ?lifetime, "TraitObject");
                let scope = Scope::TraitRefBoundary { s: self.scope };
                self.with(scope, |_, this| {
                    for bound in bounds {
                        this.visit_poly_trait_ref(bound, hir::TraitBoundModifier::None);
                    }
                });
                match lifetime.name {
                    LifetimeName::Implicit => {
                        // For types like `dyn Foo`, we should
                        // generate a special form of elided.
                        span_bug!(ty.span, "object-lifetime-default expected, not implicit",);
                    }
                    LifetimeName::ImplicitObjectLifetimeDefault => {
                        // If the user does not write *anything*, we
                        // use the object lifetime defaulting
                        // rules. So e.g., `Box<dyn Debug>` becomes
                        // `Box<dyn Debug + 'static>`.
                        self.resolve_object_lifetime_default(lifetime)
                    }
                    LifetimeName::Underscore => {
                        // If the user writes `'_`, we use the *ordinary* elision
                        // rules. So the `'_` in e.g., `Box<dyn Debug + '_>` will be
                        // resolved the same as the `'_` in `&'_ Foo`.
                        //
                        // cc #48468
                        self.resolve_elided_lifetimes(&[lifetime])
                    }
                    LifetimeName::Param(_) | LifetimeName::Static => {
                        // If the user wrote an explicit name, use that.
                        self.visit_lifetime(lifetime);
                    }
                    LifetimeName::Error => {}
                }
            }
            hir::TyKind::Rptr(ref lifetime_ref, ref mt) => {
                self.visit_lifetime(lifetime_ref);
                let scope = Scope::ObjectLifetimeDefault {
                    lifetime: self.map.defs.get(&lifetime_ref.hir_id).cloned(),
                    s: self.scope,
                };
                self.with(scope, |_, this| this.visit_ty(&mt.ty));
            }
            hir::TyKind::OpaqueDef(item_id, lifetimes) => {
                // Resolve the lifetimes in the bounds to the lifetime defs in the generics.
                // `fn foo<'a>() -> impl MyTrait<'a> { ... }` desugars to
                // `type MyAnonTy<'b> = impl MyTrait<'b>;`
                //                 ^                  ^ this gets resolved in the scope of
                //                                      the opaque_ty generics
                let opaque_ty = self.tcx.hir().item(item_id);
                let (generics, bounds) = match opaque_ty.kind {
                    // Named opaque `impl Trait` types are reached via `TyKind::Path`.
                    // This arm is for `impl Trait` in the types of statics, constants and locals.
                    hir::ItemKind::OpaqueTy(hir::OpaqueTy { impl_trait_fn: None, .. }) => {
                        intravisit::walk_ty(self, ty);

                        // Elided lifetimes are not allowed in non-return
                        // position impl Trait
                        let scope = Scope::TraitRefBoundary { s: self.scope };
                        self.with(scope, |_, this| {
                            let scope = Scope::Elision { elide: Elide::Forbid, s: this.scope };
                            this.with(scope, |_, this| {
                                intravisit::walk_item(this, opaque_ty);
                            })
                        });

                        return;
                    }
                    // RPIT (return position impl trait)
                    hir::ItemKind::OpaqueTy(hir::OpaqueTy {
                        impl_trait_fn: Some(_),
                        ref generics,
                        bounds,
                        ..
                    }) => (generics, bounds),
                    ref i => bug!("`impl Trait` pointed to non-opaque type?? {:#?}", i),
                };

                // Resolve the lifetimes that are applied to the opaque type.
                // These are resolved in the current scope.
                // `fn foo<'a>() -> impl MyTrait<'a> { ... }` desugars to
                // `fn foo<'a>() -> MyAnonTy<'a> { ... }`
                //          ^                 ^this gets resolved in the current scope
                for lifetime in lifetimes {
                    if let hir::GenericArg::Lifetime(lifetime) = lifetime {
                        self.visit_lifetime(lifetime);

                        // Check for predicates like `impl for<'a> Trait<impl OtherTrait<'a>>`
                        // and ban them. Type variables instantiated inside binders aren't
                        // well-supported at the moment, so this doesn't work.
                        // In the future, this should be fixed and this error should be removed.
                        let def = self.map.defs.get(&lifetime.hir_id).cloned();
                        if let Some(Region::LateBound(_, _, def_id, _)) = def {
                            if let Some(def_id) = def_id.as_local() {
                                let hir_id = self.tcx.hir().local_def_id_to_hir_id(def_id);
                                // Ensure that the parent of the def is an item, not HRTB
                                let parent_id = self.tcx.hir().get_parent_node(hir_id);
                                // FIXME(cjgillot) Can this check be replaced by
                                // `let parent_is_item = parent_id.is_owner();`?
                                let parent_is_item =
                                    if let Some(parent_def_id) = parent_id.as_owner() {
                                        matches!(
                                            self.tcx.hir().krate().owners.get(parent_def_id),
                                            Some(Some(_)),
                                        )
                                    } else {
                                        false
                                    };

                                if !parent_is_item {
                                    if !self.trait_definition_only {
                                        struct_span_err!(
                                            self.tcx.sess,
                                            lifetime.span,
                                            E0657,
                                            "`impl Trait` can only capture lifetimes \
                                                bound at the fn or impl level"
                                        )
                                        .emit();
                                    }
                                    self.uninsert_lifetime_on_error(lifetime, def.unwrap());
                                }
                            }
                        }
                    }
                }

                // We want to start our early-bound indices at the end of the parent scope,
                // not including any parent `impl Trait`s.
                let mut index = self.next_early_index_for_opaque_type();
                debug!(?index);

                let mut elision = None;
                let mut lifetimes = FxIndexMap::default();
                let mut non_lifetime_count = 0;
                for param in generics.params {
                    match param.kind {
                        GenericParamKind::Lifetime { .. } => {
                            let (name, reg) = Region::early(&self.tcx.hir(), &mut index, &param);
                            let def_id = if let Region::EarlyBound(_, def_id, _) = reg {
                                def_id
                            } else {
                                bug!();
                            };
                            // We cannot predict what lifetimes are unused in opaque type.
                            self.lifetime_uses.insert(def_id, LifetimeUseSet::Many);
                            if let hir::ParamName::Plain(Ident {
                                name: kw::UnderscoreLifetime,
                                ..
                            }) = name
                            {
                                // Pick the elided lifetime "definition" if one exists
                                // and use it to make an elision scope.
                                elision = Some(reg);
                            } else {
                                lifetimes.insert(name, reg);
                            }
                        }
                        GenericParamKind::Type { .. } | GenericParamKind::Const { .. } => {
                            non_lifetime_count += 1;
                        }
                    }
                }
                let next_early_index = index + non_lifetime_count;
                self.map.late_bound_vars.insert(ty.hir_id, vec![]);

                if let Some(elision_region) = elision {
                    let scope =
                        Scope::Elision { elide: Elide::Exact(elision_region), s: self.scope };
                    self.with(scope, |_old_scope, this| {
                        let scope = Scope::Binder {
                            hir_id: ty.hir_id,
                            lifetimes,
                            next_early_index,
                            s: this.scope,
                            track_lifetime_uses: true,
                            opaque_type_parent: false,
                            scope_type: BinderScopeType::Normal,
                        };
                        this.with(scope, |_old_scope, this| {
                            this.visit_generics(generics);
                            let scope = Scope::TraitRefBoundary { s: this.scope };
                            this.with(scope, |_, this| {
                                for bound in bounds {
                                    this.visit_param_bound(bound);
                                }
                            })
                        });
                    });
                } else {
                    let scope = Scope::Binder {
                        hir_id: ty.hir_id,
                        lifetimes,
                        next_early_index,
                        s: self.scope,
                        track_lifetime_uses: true,
                        opaque_type_parent: false,
                        scope_type: BinderScopeType::Normal,
                    };
                    self.with(scope, |_old_scope, this| {
                        let scope = Scope::TraitRefBoundary { s: this.scope };
                        this.with(scope, |_, this| {
                            this.visit_generics(generics);
                            for bound in bounds {
                                this.visit_param_bound(bound);
                            }
                        })
                    });
                }
            }
            _ => intravisit::walk_ty(self, ty),
        }
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem<'tcx>) {
        use self::hir::TraitItemKind::*;
        match trait_item.kind {
            Fn(ref sig, _) => {
                self.missing_named_lifetime_spots.push((&trait_item.generics).into());
                let tcx = self.tcx;
                self.visit_early_late(
                    Some(tcx.hir().get_parent_item(trait_item.hir_id())),
                    trait_item.hir_id(),
                    &sig.decl,
                    &trait_item.generics,
                    |this| intravisit::walk_trait_item(this, trait_item),
                );
                self.missing_named_lifetime_spots.pop();
            }
            Type(bounds, ref ty) => {
                self.missing_named_lifetime_spots.push((&trait_item.generics).into());
                let generics = &trait_item.generics;
                let mut index = self.next_early_index();
                debug!("visit_ty: index = {}", index);
                let mut non_lifetime_count = 0;
                let lifetimes = generics
                    .params
                    .iter()
                    .filter_map(|param| match param.kind {
                        GenericParamKind::Lifetime { .. } => {
                            Some(Region::early(&self.tcx.hir(), &mut index, param))
                        }
                        GenericParamKind::Type { .. } | GenericParamKind::Const { .. } => {
                            non_lifetime_count += 1;
                            None
                        }
                    })
                    .collect();
                self.map.late_bound_vars.insert(trait_item.hir_id(), vec![]);
                let scope = Scope::Binder {
                    hir_id: trait_item.hir_id(),
                    lifetimes,
                    next_early_index: index + non_lifetime_count,
                    s: self.scope,
                    track_lifetime_uses: true,
                    opaque_type_parent: true,
                    scope_type: BinderScopeType::Normal,
                };
                self.with(scope, |old_scope, this| {
                    this.check_lifetime_params(old_scope, &generics.params);
                    let scope = Scope::TraitRefBoundary { s: this.scope };
                    this.with(scope, |_, this| {
                        this.visit_generics(generics);
                        for bound in bounds {
                            this.visit_param_bound(bound);
                        }
                        if let Some(ty) = ty {
                            this.visit_ty(ty);
                        }
                    })
                });
                self.missing_named_lifetime_spots.pop();
            }
            Const(_, _) => {
                // Only methods and types support generics.
                assert!(trait_item.generics.params.is_empty());
                self.missing_named_lifetime_spots.push(MissingLifetimeSpot::Static);
                intravisit::walk_trait_item(self, trait_item);
                self.missing_named_lifetime_spots.pop();
            }
        }
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem<'tcx>) {
        use self::hir::ImplItemKind::*;
        match impl_item.kind {
            Fn(ref sig, _) => {
                self.missing_named_lifetime_spots.push((&impl_item.generics).into());
                let tcx = self.tcx;
                self.visit_early_late(
                    Some(tcx.hir().get_parent_item(impl_item.hir_id())),
                    impl_item.hir_id(),
                    &sig.decl,
                    &impl_item.generics,
                    |this| intravisit::walk_impl_item(this, impl_item),
                );
                self.missing_named_lifetime_spots.pop();
            }
            TyAlias(ref ty) => {
                let generics = &impl_item.generics;
                self.missing_named_lifetime_spots.push(generics.into());
                let mut index = self.next_early_index();
                let mut non_lifetime_count = 0;
                debug!("visit_ty: index = {}", index);
                let lifetimes: FxIndexMap<hir::ParamName, Region> = generics
                    .params
                    .iter()
                    .filter_map(|param| match param.kind {
                        GenericParamKind::Lifetime { .. } => {
                            Some(Region::early(&self.tcx.hir(), &mut index, param))
                        }
                        GenericParamKind::Const { .. } | GenericParamKind::Type { .. } => {
                            non_lifetime_count += 1;
                            None
                        }
                    })
                    .collect();
                self.map.late_bound_vars.insert(ty.hir_id, vec![]);
                let scope = Scope::Binder {
                    hir_id: ty.hir_id,
                    lifetimes,
                    next_early_index: index + non_lifetime_count,
                    s: self.scope,
                    track_lifetime_uses: true,
                    opaque_type_parent: true,
                    scope_type: BinderScopeType::Normal,
                };
                self.with(scope, |old_scope, this| {
                    this.check_lifetime_params(old_scope, &generics.params);
                    let scope = Scope::TraitRefBoundary { s: this.scope };
                    this.with(scope, |_, this| {
                        this.visit_generics(generics);
                        this.visit_ty(ty);
                    })
                });
                self.missing_named_lifetime_spots.pop();
            }
            Const(_, _) => {
                // Only methods and types support generics.
                assert!(impl_item.generics.params.is_empty());
                self.missing_named_lifetime_spots.push(MissingLifetimeSpot::Static);
                intravisit::walk_impl_item(self, impl_item);
                self.missing_named_lifetime_spots.pop();
            }
        }
    }

    #[tracing::instrument(level = "debug", skip(self))]
    fn visit_lifetime(&mut self, lifetime_ref: &'tcx hir::Lifetime) {
        if lifetime_ref.is_elided() {
            self.resolve_elided_lifetimes(&[lifetime_ref]);
            return;
        }
        if lifetime_ref.is_static() {
            self.insert_lifetime(lifetime_ref, Region::Static);
            return;
        }
        if self.is_in_const_generic && lifetime_ref.name != LifetimeName::Error {
            self.emit_non_static_lt_in_const_generic_error(lifetime_ref);
            return;
        }
        self.resolve_lifetime_ref(lifetime_ref);
    }

    fn visit_assoc_type_binding(&mut self, type_binding: &'tcx hir::TypeBinding<'_>) {
        let scope = self.scope;
        if let Some(scope_for_path) = self.map.scope_for_path.as_mut() {
            // We add lifetime scope information for `Ident`s in associated type bindings and use
            // the `HirId` of the type binding as the key in `LifetimeMap`
            let lifetime_scope = get_lifetime_scopes_for_path(scope);
            let map = scope_for_path.entry(type_binding.hir_id.owner).or_default();
            map.insert(type_binding.hir_id.local_id, lifetime_scope);
        }
        hir::intravisit::walk_assoc_type_binding(self, type_binding);
    }

    fn visit_path(&mut self, path: &'tcx hir::Path<'tcx>, _: hir::HirId) {
        for (i, segment) in path.segments.iter().enumerate() {
            let depth = path.segments.len() - i - 1;
            if let Some(ref args) = segment.args {
                self.visit_segment_args(path.res, depth, args);
            }

            let scope = self.scope;
            if let Some(scope_for_path) = self.map.scope_for_path.as_mut() {
                // Add lifetime scope information to path segment. Note we cannot call `visit_path_segment`
                // here because that call would yield to resolution problems due to `walk_path_segment`
                // being called, which processes the path segments generic args, which we have already
                // processed using `visit_segment_args`.
                let lifetime_scope = get_lifetime_scopes_for_path(scope);
                if let Some(hir_id) = segment.hir_id {
                    let map = scope_for_path.entry(hir_id.owner).or_default();
                    map.insert(hir_id.local_id, lifetime_scope);
                }
            }
        }
    }

    fn visit_path_segment(&mut self, path_span: Span, path_segment: &'tcx hir::PathSegment<'tcx>) {
        let scope = self.scope;
        if let Some(scope_for_path) = self.map.scope_for_path.as_mut() {
            let lifetime_scope = get_lifetime_scopes_for_path(scope);
            if let Some(hir_id) = path_segment.hir_id {
                let map = scope_for_path.entry(hir_id.owner).or_default();
                map.insert(hir_id.local_id, lifetime_scope);
            }
        }

        intravisit::walk_path_segment(self, path_span, path_segment);
    }

    fn visit_fn_decl(&mut self, fd: &'tcx hir::FnDecl<'tcx>) {
        let output = match fd.output {
            hir::FnRetTy::DefaultReturn(_) => None,
            hir::FnRetTy::Return(ref ty) => Some(&**ty),
        };
        self.visit_fn_like_elision(&fd.inputs, output);
    }

    fn visit_generics(&mut self, generics: &'tcx hir::Generics<'tcx>) {
        if !self.trait_definition_only {
            check_mixed_explicit_and_in_band_defs(self.tcx, &generics.params);
        }
        let scope = Scope::TraitRefBoundary { s: self.scope };
        self.with(scope, |_, this| {
            for param in generics.params {
                match param.kind {
                    GenericParamKind::Lifetime { .. } => {}
                    GenericParamKind::Type { ref default, .. } => {
                        walk_list!(this, visit_param_bound, param.bounds);
                        if let Some(ref ty) = default {
                            this.visit_ty(&ty);
                        }
                    }
                    GenericParamKind::Const { ref ty, .. } => {
                        let was_in_const_generic = this.is_in_const_generic;
                        this.is_in_const_generic = true;
                        walk_list!(this, visit_param_bound, param.bounds);
                        this.visit_ty(&ty);
                        this.is_in_const_generic = was_in_const_generic;
                    }
                }
            }
            for predicate in generics.where_clause.predicates {
                match predicate {
                    &hir::WherePredicate::BoundPredicate(hir::WhereBoundPredicate {
                        ref bounded_ty,
                        bounds,
                        ref bound_generic_params,
                        ..
                    }) => {
                        let (lifetimes, binders): (FxIndexMap<hir::ParamName, Region>, Vec<_>) =
                            bound_generic_params
                                .iter()
                                .filter_map(|param| match param.kind {
                                    GenericParamKind::Lifetime { .. } => Some(param),
                                    _ => None,
                                })
                                .enumerate()
                                .map(|(late_bound_idx, param)| {
                                    let pair =
                                        Region::late(late_bound_idx as u32, &this.tcx.hir(), param);
                                    let r = late_region_as_bound_region(this.tcx, &pair.1);
                                    (pair, r)
                                })
                                .unzip();
                        this.map.late_bound_vars.insert(bounded_ty.hir_id, binders.clone());
                        let next_early_index = this.next_early_index();
                        // Even if there are no lifetimes defined here, we still wrap it in a binder
                        // scope. If there happens to be a nested poly trait ref (an error), that
                        // will be `Concatenating` anyways, so we don't have to worry about the depth
                        // being wrong.
                        let scope = Scope::Binder {
                            hir_id: bounded_ty.hir_id,
                            lifetimes,
                            s: this.scope,
                            next_early_index,
                            track_lifetime_uses: true,
                            opaque_type_parent: false,
                            scope_type: BinderScopeType::Normal,
                        };
                        this.with(scope, |old_scope, this| {
                            this.check_lifetime_params(old_scope, &bound_generic_params);
                            this.visit_ty(&bounded_ty);
                            walk_list!(this, visit_param_bound, bounds);
                        })
                    }
                    &hir::WherePredicate::RegionPredicate(hir::WhereRegionPredicate {
                        ref lifetime,
                        bounds,
                        ..
                    }) => {
                        this.visit_lifetime(lifetime);
                        walk_list!(this, visit_param_bound, bounds);
                    }
                    &hir::WherePredicate::EqPredicate(hir::WhereEqPredicate {
                        ref lhs_ty,
                        ref rhs_ty,
                        ..
                    }) => {
                        this.visit_ty(lhs_ty);
                        this.visit_ty(rhs_ty);
                    }
                }
            }
        })
    }

    fn visit_param_bound(&mut self, bound: &'tcx hir::GenericBound<'tcx>) {
        match bound {
            hir::GenericBound::LangItemTrait(_, _, hir_id, _) => {
                // FIXME(jackh726): This is pretty weird. `LangItemTrait` doesn't go
                // through the regular poly trait ref code, so we don't get another
                // chance to introduce a binder. For now, I'm keeping the existing logic
                // of "if there isn't a Binder scope above us, add one", but I
                // imagine there's a better way to go about this.
                let (binders, scope_type) = self.poly_trait_ref_binder_info();

                self.map.late_bound_vars.insert(*hir_id, binders);
                let scope = Scope::Binder {
                    hir_id: *hir_id,
                    lifetimes: FxIndexMap::default(),
                    s: self.scope,
                    next_early_index: self.next_early_index(),
                    track_lifetime_uses: true,
                    opaque_type_parent: false,
                    scope_type,
                };
                self.with(scope, |_, this| {
                    intravisit::walk_param_bound(this, bound);
                });
            }
            _ => intravisit::walk_param_bound(self, bound),
        }
    }

    fn visit_poly_trait_ref(
        &mut self,
        trait_ref: &'tcx hir::PolyTraitRef<'tcx>,
        _modifier: hir::TraitBoundModifier,
    ) {
        debug!("visit_poly_trait_ref(trait_ref={:?})", trait_ref);

        let should_pop_missing_lt = self.is_trait_ref_fn_scope(trait_ref);

        let next_early_index = self.next_early_index();
        let (mut binders, scope_type) = self.poly_trait_ref_binder_info();

        let initial_bound_vars = binders.len() as u32;
        let mut lifetimes: FxIndexMap<hir::ParamName, Region> = FxIndexMap::default();
        let binders_iter = trait_ref
            .bound_generic_params
            .iter()
            .filter_map(|param| match param.kind {
                GenericParamKind::Lifetime { .. } => Some(param),
                _ => None,
            })
            .enumerate()
            .map(|(late_bound_idx, param)| {
                let pair = Region::late(
                    initial_bound_vars + late_bound_idx as u32,
                    &self.tcx.hir(),
                    param,
                );
                let r = late_region_as_bound_region(self.tcx, &pair.1);
                lifetimes.insert(pair.0, pair.1);
                r
            });
        binders.extend(binders_iter);

        debug!(?binders);
        self.map.late_bound_vars.insert(trait_ref.trait_ref.hir_ref_id, binders);

        // Always introduce a scope here, even if this is in a where clause and
        // we introduced the binders around the bounded Ty. In that case, we
        // just reuse the concatenation functionality also present in nested trait
        // refs.
        let scope = Scope::Binder {
            hir_id: trait_ref.trait_ref.hir_ref_id,
            lifetimes,
            s: self.scope,
            next_early_index,
            track_lifetime_uses: true,
            opaque_type_parent: false,
            scope_type,
        };
        self.with(scope, |old_scope, this| {
            this.check_lifetime_params(old_scope, &trait_ref.bound_generic_params);
            walk_list!(this, visit_generic_param, trait_ref.bound_generic_params);
            this.visit_trait_ref(&trait_ref.trait_ref);
        });

        if should_pop_missing_lt {
            self.missing_named_lifetime_spots.pop();
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
enum ShadowKind {
    Label,
    Lifetime,
}
struct Original {
    kind: ShadowKind,
    span: Span,
}
struct Shadower {
    kind: ShadowKind,
    span: Span,
}

fn original_label(span: Span) -> Original {
    Original { kind: ShadowKind::Label, span }
}
fn shadower_label(span: Span) -> Shadower {
    Shadower { kind: ShadowKind::Label, span }
}
fn original_lifetime(span: Span) -> Original {
    Original { kind: ShadowKind::Lifetime, span }
}
fn shadower_lifetime(param: &hir::GenericParam<'_>) -> Shadower {
    Shadower { kind: ShadowKind::Lifetime, span: param.span }
}

impl ShadowKind {
    fn desc(&self) -> &'static str {
        match *self {
            ShadowKind::Label => "label",
            ShadowKind::Lifetime => "lifetime",
        }
    }
}

fn check_mixed_explicit_and_in_band_defs(tcx: TyCtxt<'_>, params: &[hir::GenericParam<'_>]) {
    let lifetime_params: Vec<_> = params
        .iter()
        .filter_map(|param| match param.kind {
            GenericParamKind::Lifetime { kind, .. } => Some((kind, param.span)),
            _ => None,
        })
        .collect();
    let explicit = lifetime_params.iter().find(|(kind, _)| *kind == LifetimeParamKind::Explicit);
    let in_band = lifetime_params.iter().find(|(kind, _)| *kind == LifetimeParamKind::InBand);

    if let (Some((_, explicit_span)), Some((_, in_band_span))) = (explicit, in_band) {
        struct_span_err!(
            tcx.sess,
            *in_band_span,
            E0688,
            "cannot mix in-band and explicit lifetime definitions"
        )
        .span_label(*in_band_span, "in-band lifetime definition here")
        .span_label(*explicit_span, "explicit lifetime definition here")
        .emit();
    }
}

fn signal_shadowing_problem(tcx: TyCtxt<'_>, name: Symbol, orig: Original, shadower: Shadower) {
    let mut err = if let (ShadowKind::Lifetime, ShadowKind::Lifetime) = (orig.kind, shadower.kind) {
        // lifetime/lifetime shadowing is an error
        struct_span_err!(
            tcx.sess,
            shadower.span,
            E0496,
            "{} name `{}` shadows a \
             {} name that is already in scope",
            shadower.kind.desc(),
            name,
            orig.kind.desc()
        )
    } else {
        // shadowing involving a label is only a warning, due to issues with
        // labels and lifetimes not being macro-hygienic.
        tcx.sess.struct_span_warn(
            shadower.span,
            &format!(
                "{} name `{}` shadows a \
                 {} name that is already in scope",
                shadower.kind.desc(),
                name,
                orig.kind.desc()
            ),
        )
    };
    err.span_label(orig.span, "first declared here");
    err.span_label(shadower.span, format!("{} `{}` already in scope", orig.kind.desc(), name));
    err.emit();
}

// Adds all labels in `b` to `ctxt.labels_in_fn`, signalling a warning
// if one of the label shadows a lifetime or another label.
fn extract_labels(ctxt: &mut LifetimeContext<'_, '_>, body: &hir::Body<'_>) {
    struct GatherLabels<'a, 'tcx> {
        tcx: TyCtxt<'tcx>,
        scope: ScopeRef<'a>,
        labels_in_fn: &'a mut Vec<Ident>,
    }

    let mut gather =
        GatherLabels { tcx: ctxt.tcx, scope: ctxt.scope, labels_in_fn: &mut ctxt.labels_in_fn };
    gather.visit_body(body);

    impl<'v, 'a, 'tcx> Visitor<'v> for GatherLabels<'a, 'tcx> {
        type Map = intravisit::ErasedMap<'v>;

        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
            NestedVisitorMap::None
        }

        fn visit_expr(&mut self, ex: &hir::Expr<'_>) {
            if let Some(label) = expression_label(ex) {
                for prior_label in &self.labels_in_fn[..] {
                    // FIXME (#24278): non-hygienic comparison
                    if label.name == prior_label.name {
                        signal_shadowing_problem(
                            self.tcx,
                            label.name,
                            original_label(prior_label.span),
                            shadower_label(label.span),
                        );
                    }
                }

                check_if_label_shadows_lifetime(self.tcx, self.scope, label);

                self.labels_in_fn.push(label);
            }
            intravisit::walk_expr(self, ex)
        }
    }

    fn expression_label(ex: &hir::Expr<'_>) -> Option<Ident> {
        if let hir::ExprKind::Loop(_, Some(label), ..) = ex.kind { Some(label.ident) } else { None }
    }

    fn check_if_label_shadows_lifetime(tcx: TyCtxt<'_>, mut scope: ScopeRef<'_>, label: Ident) {
        loop {
            match *scope {
                Scope::Body { s, .. }
                | Scope::Elision { s, .. }
                | Scope::ObjectLifetimeDefault { s, .. }
                | Scope::Supertrait { s, .. }
                | Scope::TraitRefBoundary { s, .. } => {
                    scope = s;
                }

                Scope::Root => {
                    return;
                }

                Scope::Binder { ref lifetimes, s, .. } => {
                    // FIXME (#24278): non-hygienic comparison
                    if let Some(def) =
                        lifetimes.get(&hir::ParamName::Plain(label.normalize_to_macros_2_0()))
                    {
                        let hir_id =
                            tcx.hir().local_def_id_to_hir_id(def.id().unwrap().expect_local());

                        signal_shadowing_problem(
                            tcx,
                            label.name,
                            original_lifetime(tcx.hir().span(hir_id)),
                            shadower_label(label.span),
                        );
                        return;
                    }
                    scope = s;
                }
            }
        }
    }
}

fn compute_object_lifetime_defaults(
    tcx: TyCtxt<'_>,
    item: &hir::Item<'_>,
) -> Option<Vec<ObjectLifetimeDefault>> {
    match item.kind {
        hir::ItemKind::Struct(_, ref generics)
        | hir::ItemKind::Union(_, ref generics)
        | hir::ItemKind::Enum(_, ref generics)
        | hir::ItemKind::OpaqueTy(hir::OpaqueTy { ref generics, impl_trait_fn: None, .. })
        | hir::ItemKind::TyAlias(_, ref generics)
        | hir::ItemKind::Trait(_, _, ref generics, ..) => {
            let result = object_lifetime_defaults_for_item(tcx, generics);

            // Debugging aid.
            let attrs = tcx.hir().attrs(item.hir_id());
            if tcx.sess.contains_name(attrs, sym::rustc_object_lifetime_default) {
                let object_lifetime_default_reprs: String = result
                    .iter()
                    .map(|set| match *set {
                        Set1::Empty => "BaseDefault".into(),
                        Set1::One(Region::Static) => "'static".into(),
                        Set1::One(Region::EarlyBound(mut i, _, _)) => generics
                            .params
                            .iter()
                            .find_map(|param| match param.kind {
                                GenericParamKind::Lifetime { .. } => {
                                    if i == 0 {
                                        return Some(param.name.ident().to_string().into());
                                    }
                                    i -= 1;
                                    None
                                }
                                _ => None,
                            })
                            .unwrap(),
                        Set1::One(_) => bug!(),
                        Set1::Many => "Ambiguous".into(),
                    })
                    .collect::<Vec<Cow<'static, str>>>()
                    .join(",");
                tcx.sess.span_err(item.span, &object_lifetime_default_reprs);
            }

            Some(result)
        }
        _ => None,
    }
}

/// Scan the bounds and where-clauses on parameters to extract bounds
/// of the form `T:'a` so as to determine the `ObjectLifetimeDefault`
/// for each type parameter.
fn object_lifetime_defaults_for_item(
    tcx: TyCtxt<'_>,
    generics: &hir::Generics<'_>,
) -> Vec<ObjectLifetimeDefault> {
    fn add_bounds(set: &mut Set1<hir::LifetimeName>, bounds: &[hir::GenericBound<'_>]) {
        for bound in bounds {
            if let hir::GenericBound::Outlives(ref lifetime) = *bound {
                set.insert(lifetime.name.normalize_to_macros_2_0());
            }
        }
    }

    generics
        .params
        .iter()
        .filter_map(|param| match param.kind {
            GenericParamKind::Lifetime { .. } => None,
            GenericParamKind::Type { .. } => {
                let mut set = Set1::Empty;

                add_bounds(&mut set, &param.bounds);

                let param_def_id = tcx.hir().local_def_id(param.hir_id);
                for predicate in generics.where_clause.predicates {
                    // Look for `type: ...` where clauses.
                    let data = match *predicate {
                        hir::WherePredicate::BoundPredicate(ref data) => data,
                        _ => continue,
                    };

                    // Ignore `for<'a> type: ...` as they can change what
                    // lifetimes mean (although we could "just" handle it).
                    if !data.bound_generic_params.is_empty() {
                        continue;
                    }

                    let res = match data.bounded_ty.kind {
                        hir::TyKind::Path(hir::QPath::Resolved(None, ref path)) => path.res,
                        _ => continue,
                    };

                    if res == Res::Def(DefKind::TyParam, param_def_id.to_def_id()) {
                        add_bounds(&mut set, &data.bounds);
                    }
                }

                Some(match set {
                    Set1::Empty => Set1::Empty,
                    Set1::One(name) => {
                        if name == hir::LifetimeName::Static {
                            Set1::One(Region::Static)
                        } else {
                            generics
                                .params
                                .iter()
                                .filter_map(|param| match param.kind {
                                    GenericParamKind::Lifetime { .. } => Some((
                                        param.hir_id,
                                        hir::LifetimeName::Param(param.name),
                                        LifetimeDefOrigin::from_param(param),
                                    )),
                                    _ => None,
                                })
                                .enumerate()
                                .find(|&(_, (_, lt_name, _))| lt_name == name)
                                .map_or(Set1::Many, |(i, (id, _, origin))| {
                                    let def_id = tcx.hir().local_def_id(id);
                                    Set1::One(Region::EarlyBound(
                                        i as u32,
                                        def_id.to_def_id(),
                                        origin,
                                    ))
                                })
                        }
                    }
                    Set1::Many => Set1::Many,
                })
            }
            GenericParamKind::Const { .. } => {
                // Generic consts don't impose any constraints.
                //
                // We still store a dummy value here to allow generic parameters
                // in an arbitrary order.
                Some(Set1::Empty)
            }
        })
        .collect()
}

impl<'a, 'tcx> LifetimeContext<'a, 'tcx> {
    fn with<F>(&mut self, wrap_scope: Scope<'_>, f: F)
    where
        F: for<'b> FnOnce(ScopeRef<'_>, &mut LifetimeContext<'b, 'tcx>),
    {
        let LifetimeContext { tcx, map, lifetime_uses, .. } = self;
        let labels_in_fn = take(&mut self.labels_in_fn);
        let xcrate_object_lifetime_defaults = take(&mut self.xcrate_object_lifetime_defaults);
        let missing_named_lifetime_spots = take(&mut self.missing_named_lifetime_spots);
        let mut this = LifetimeContext {
            tcx: *tcx,
            map,
            scope: &wrap_scope,
            is_in_fn_syntax: self.is_in_fn_syntax,
            is_in_const_generic: self.is_in_const_generic,
            trait_definition_only: self.trait_definition_only,
            labels_in_fn,
            xcrate_object_lifetime_defaults,
            lifetime_uses,
            missing_named_lifetime_spots,
        };
        let span = tracing::debug_span!("scope", scope = ?TruncatedScopeDebug(&this.scope));
        {
            let _enter = span.enter();
            f(self.scope, &mut this);
            if !self.trait_definition_only {
                this.check_uses_for_lifetimes_defined_by_scope();
            }
        }
        self.labels_in_fn = this.labels_in_fn;
        self.xcrate_object_lifetime_defaults = this.xcrate_object_lifetime_defaults;
        self.missing_named_lifetime_spots = this.missing_named_lifetime_spots;
    }

    /// helper method to determine the span to remove when suggesting the
    /// deletion of a lifetime
    fn lifetime_deletion_span(&self, name: Ident, generics: &hir::Generics<'_>) -> Option<Span> {
        generics.params.iter().enumerate().find_map(|(i, param)| {
            if param.name.ident() == name {
                let in_band = matches!(
                    param.kind,
                    hir::GenericParamKind::Lifetime { kind: hir::LifetimeParamKind::InBand }
                );
                if in_band {
                    Some(param.span)
                } else if generics.params.len() == 1 {
                    // if sole lifetime, remove the entire `<>` brackets
                    Some(generics.span)
                } else {
                    // if removing within `<>` brackets, we also want to
                    // delete a leading or trailing comma as appropriate
                    if i >= generics.params.len() - 1 {
                        Some(generics.params[i - 1].span.shrink_to_hi().to(param.span))
                    } else {
                        Some(param.span.to(generics.params[i + 1].span.shrink_to_lo()))
                    }
                }
            } else {
                None
            }
        })
    }

    // helper method to issue suggestions from `fn rah<'a>(&'a T)` to `fn rah(&T)`
    // or from `fn rah<'a>(T<'a>)` to `fn rah(T<'_>)`
    fn suggest_eliding_single_use_lifetime(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        def_id: DefId,
        lifetime: &hir::Lifetime,
    ) {
        let name = lifetime.name.ident();
        let remove_decl = self
            .tcx
            .parent(def_id)
            .and_then(|parent_def_id| self.tcx.hir().get_generics(parent_def_id))
            .and_then(|generics| self.lifetime_deletion_span(name, generics));

        let mut remove_use = None;
        let mut elide_use = None;
        let mut find_arg_use_span = |inputs: &[hir::Ty<'_>]| {
            for input in inputs {
                match input.kind {
                    hir::TyKind::Rptr(lt, _) => {
                        if lt.name.ident() == name {
                            // include the trailing whitespace between the lifetime and type names
                            let lt_through_ty_span = lifetime.span.to(input.span.shrink_to_hi());
                            remove_use = Some(
                                self.tcx
                                    .sess
                                    .source_map()
                                    .span_until_non_whitespace(lt_through_ty_span),
                            );
                            break;
                        }
                    }
                    hir::TyKind::Path(ref qpath) => {
                        if let QPath::Resolved(_, path) = qpath {
                            let last_segment = &path.segments[path.segments.len() - 1];
                            let generics = last_segment.args();
                            for arg in generics.args.iter() {
                                if let GenericArg::Lifetime(lt) = arg {
                                    if lt.name.ident() == name {
                                        elide_use = Some(lt.span);
                                        break;
                                    }
                                }
                            }
                            break;
                        }
                    }
                    _ => {}
                }
            }
        };
        if let Node::Lifetime(hir_lifetime) = self.tcx.hir().get(lifetime.hir_id) {
            if let Some(parent) =
                self.tcx.hir().find(self.tcx.hir().get_parent_item(hir_lifetime.hir_id))
            {
                match parent {
                    Node::Item(item) => {
                        if let hir::ItemKind::Fn(sig, _, _) = &item.kind {
                            find_arg_use_span(sig.decl.inputs);
                        }
                    }
                    Node::ImplItem(impl_item) => {
                        if let hir::ImplItemKind::Fn(sig, _) = &impl_item.kind {
                            find_arg_use_span(sig.decl.inputs);
                        }
                    }
                    _ => {}
                }
            }
        }

        let msg = "elide the single-use lifetime";
        match (remove_decl, remove_use, elide_use) {
            (Some(decl_span), Some(use_span), None) => {
                // if both declaration and use deletion spans start at the same
                // place ("start at" because the latter includes trailing
                // whitespace), then this is an in-band lifetime
                if decl_span.shrink_to_lo() == use_span.shrink_to_lo() {
                    err.span_suggestion(
                        use_span,
                        msg,
                        String::new(),
                        Applicability::MachineApplicable,
                    );
                } else {
                    err.multipart_suggestion(
                        msg,
                        vec![(decl_span, String::new()), (use_span, String::new())],
                        Applicability::MachineApplicable,
                    );
                }
            }
            (Some(decl_span), None, Some(use_span)) => {
                err.multipart_suggestion(
                    msg,
                    vec![(decl_span, String::new()), (use_span, "'_".to_owned())],
                    Applicability::MachineApplicable,
                );
            }
            _ => {}
        }
    }

    fn check_uses_for_lifetimes_defined_by_scope(&mut self) {
        let defined_by = match self.scope {
            Scope::Binder { lifetimes, .. } => lifetimes,
            _ => {
                debug!("check_uses_for_lifetimes_defined_by_scope: not in a binder scope");
                return;
            }
        };

        let mut def_ids: Vec<_> = defined_by
            .values()
            .flat_map(|region| match region {
                Region::EarlyBound(_, def_id, _)
                | Region::LateBound(_, _, def_id, _)
                | Region::Free(_, def_id) => Some(*def_id),

                Region::LateBoundAnon(..) | Region::Static => None,
            })
            .collect();

        // ensure that we issue lints in a repeatable order
        def_ids.sort_by_cached_key(|&def_id| self.tcx.def_path_hash(def_id));

        for def_id in def_ids {
            debug!("check_uses_for_lifetimes_defined_by_scope: def_id = {:?}", def_id);

            let lifetimeuseset = self.lifetime_uses.remove(&def_id);

            debug!(
                "check_uses_for_lifetimes_defined_by_scope: lifetimeuseset = {:?}",
                lifetimeuseset
            );

            match lifetimeuseset {
                Some(LifetimeUseSet::One(lifetime)) => {
                    let hir_id = self.tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
                    debug!("hir id first={:?}", hir_id);
                    if let Some((id, span, name)) = match self.tcx.hir().get(hir_id) {
                        Node::Lifetime(hir_lifetime) => Some((
                            hir_lifetime.hir_id,
                            hir_lifetime.span,
                            hir_lifetime.name.ident(),
                        )),
                        Node::GenericParam(param) => {
                            Some((param.hir_id, param.span, param.name.ident()))
                        }
                        _ => None,
                    } {
                        debug!("id = {:?} span = {:?} name = {:?}", id, span, name);
                        if name.name == kw::UnderscoreLifetime {
                            continue;
                        }

                        if let Some(parent_def_id) = self.tcx.parent(def_id) {
                            if let Some(def_id) = parent_def_id.as_local() {
                                let parent_hir_id = self.tcx.hir().local_def_id_to_hir_id(def_id);
                                // lifetimes in `derive` expansions don't count (Issue #53738)
                                if self
                                    .tcx
                                    .hir()
                                    .attrs(parent_hir_id)
                                    .iter()
                                    .any(|attr| attr.has_name(sym::automatically_derived))
                                {
                                    continue;
                                }
                            }
                        }

                        self.tcx.struct_span_lint_hir(
                            lint::builtin::SINGLE_USE_LIFETIMES,
                            id,
                            span,
                            |lint| {
                                let mut err = lint.build(&format!(
                                    "lifetime parameter `{}` only used once",
                                    name
                                ));
                                if span == lifetime.span {
                                    // spans are the same for in-band lifetime declarations
                                    err.span_label(span, "this lifetime is only used here");
                                } else {
                                    err.span_label(span, "this lifetime...");
                                    err.span_label(lifetime.span, "...is used only here");
                                }
                                self.suggest_eliding_single_use_lifetime(
                                    &mut err, def_id, lifetime,
                                );
                                err.emit();
                            },
                        );
                    }
                }
                Some(LifetimeUseSet::Many) => {
                    debug!("not one use lifetime");
                }
                None => {
                    let hir_id = self.tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
                    if let Some((id, span, name)) = match self.tcx.hir().get(hir_id) {
                        Node::Lifetime(hir_lifetime) => Some((
                            hir_lifetime.hir_id,
                            hir_lifetime.span,
                            hir_lifetime.name.ident(),
                        )),
                        Node::GenericParam(param) => {
                            Some((param.hir_id, param.span, param.name.ident()))
                        }
                        _ => None,
                    } {
                        debug!("id ={:?} span = {:?} name = {:?}", id, span, name);
                        self.tcx.struct_span_lint_hir(
                            lint::builtin::UNUSED_LIFETIMES,
                            id,
                            span,
                            |lint| {
                                let mut err = lint
                                    .build(&format!("lifetime parameter `{}` never used", name));
                                if let Some(parent_def_id) = self.tcx.parent(def_id) {
                                    if let Some(generics) =
                                        self.tcx.hir().get_generics(parent_def_id)
                                    {
                                        let unused_lt_span =
                                            self.lifetime_deletion_span(name, generics);
                                        if let Some(span) = unused_lt_span {
                                            err.span_suggestion(
                                                span,
                                                "elide the unused lifetime",
                                                String::new(),
                                                Applicability::MachineApplicable,
                                            );
                                        }
                                    }
                                }
                                err.emit();
                            },
                        );
                    }
                }
            }
        }
    }

    /// Visits self by adding a scope and handling recursive walk over the contents with `walk`.
    ///
    /// Handles visiting fns and methods. These are a bit complicated because we must distinguish
    /// early- vs late-bound lifetime parameters. We do this by checking which lifetimes appear
    /// within type bounds; those are early bound lifetimes, and the rest are late bound.
    ///
    /// For example:
    ///
    ///    fn foo<'a,'b,'c,T:Trait<'b>>(...)
    ///
    /// Here `'a` and `'c` are late bound but `'b` is early bound. Note that early- and late-bound
    /// lifetimes may be interspersed together.
    ///
    /// If early bound lifetimes are present, we separate them into their own list (and likewise
    /// for late bound). They will be numbered sequentially, starting from the lowest index that is
    /// already in scope (for a fn item, that will be 0, but for a method it might not be). Late
    /// bound lifetimes are resolved by name and associated with a binder ID (`binder_id`), so the
    /// ordering is not important there.
    fn visit_early_late<F>(
        &mut self,
        parent_id: Option<hir::HirId>,
        hir_id: hir::HirId,
        decl: &'tcx hir::FnDecl<'tcx>,
        generics: &'tcx hir::Generics<'tcx>,
        walk: F,
    ) where
        F: for<'b, 'c> FnOnce(&'b mut LifetimeContext<'c, 'tcx>),
    {
        insert_late_bound_lifetimes(self.map, decl, generics);

        // Find the start of nested early scopes, e.g., in methods.
        let mut next_early_index = 0;
        if let Some(parent_id) = parent_id {
            let parent = self.tcx.hir().expect_item(parent_id);
            if sub_items_have_self_param(&parent.kind) {
                next_early_index += 1; // Self comes before lifetimes
            }
            match parent.kind {
                hir::ItemKind::Trait(_, _, ref generics, ..)
                | hir::ItemKind::Impl(hir::Impl { ref generics, .. }) => {
                    next_early_index += generics.params.len() as u32;
                }
                _ => {}
            }
        }

        let mut non_lifetime_count = 0;
        let mut named_late_bound_vars = 0;
        let lifetimes: FxIndexMap<hir::ParamName, Region> = generics
            .params
            .iter()
            .filter_map(|param| match param.kind {
                GenericParamKind::Lifetime { .. } => {
                    if self.map.late_bound.contains(&param.hir_id) {
                        let late_bound_idx = named_late_bound_vars;
                        named_late_bound_vars += 1;
                        Some(Region::late(late_bound_idx, &self.tcx.hir(), param))
                    } else {
                        Some(Region::early(&self.tcx.hir(), &mut next_early_index, param))
                    }
                }
                GenericParamKind::Type { .. } | GenericParamKind::Const { .. } => {
                    non_lifetime_count += 1;
                    None
                }
            })
            .collect();
        let next_early_index = next_early_index + non_lifetime_count;

        let binders: Vec<_> = generics
            .params
            .iter()
            .filter_map(|param| match param.kind {
                GenericParamKind::Lifetime { .. }
                    if self.map.late_bound.contains(&param.hir_id) =>
                {
                    Some(param)
                }
                _ => None,
            })
            .enumerate()
            .map(|(late_bound_idx, param)| {
                let pair = Region::late(late_bound_idx as u32, &self.tcx.hir(), param);
                let r = late_region_as_bound_region(self.tcx, &pair.1);
                r
            })
            .collect();
        self.map.late_bound_vars.insert(hir_id, binders);
        let scope = Scope::Binder {
            hir_id,
            lifetimes,
            next_early_index,
            s: self.scope,
            opaque_type_parent: true,
            track_lifetime_uses: false,
            scope_type: BinderScopeType::Normal,
        };
        self.with(scope, move |old_scope, this| {
            this.check_lifetime_params(old_scope, &generics.params);
            walk(this);
        });
    }

    fn next_early_index_helper(&self, only_opaque_type_parent: bool) -> u32 {
        let mut scope = self.scope;
        loop {
            match *scope {
                Scope::Root => return 0,

                Scope::Binder { next_early_index, opaque_type_parent, .. }
                    if (!only_opaque_type_parent || opaque_type_parent) =>
                {
                    return next_early_index;
                }

                Scope::Binder { s, .. }
                | Scope::Body { s, .. }
                | Scope::Elision { s, .. }
                | Scope::ObjectLifetimeDefault { s, .. }
                | Scope::Supertrait { s, .. }
                | Scope::TraitRefBoundary { s, .. } => scope = s,
            }
        }
    }

    /// Returns the next index one would use for an early-bound-region
    /// if extending the current scope.
    fn next_early_index(&self) -> u32 {
        self.next_early_index_helper(true)
    }

    /// Returns the next index one would use for an `impl Trait` that
    /// is being converted into an opaque type alias `impl Trait`. This will be the
    /// next early index from the enclosing item, for the most
    /// part. See the `opaque_type_parent` field for more info.
    fn next_early_index_for_opaque_type(&self) -> u32 {
        self.next_early_index_helper(false)
    }

    fn resolve_lifetime_ref(&mut self, lifetime_ref: &'tcx hir::Lifetime) {
        debug!("resolve_lifetime_ref(lifetime_ref={:?})", lifetime_ref);

        // If we've already reported an error, just ignore `lifetime_ref`.
        if let LifetimeName::Error = lifetime_ref.name {
            return;
        }

        // Walk up the scope chain, tracking the number of fn scopes
        // that we pass through, until we find a lifetime with the
        // given name or we run out of scopes.
        // search.
        let mut late_depth = 0;
        let mut scope = self.scope;
        let mut outermost_body = None;
        let result = loop {
            match *scope {
                Scope::Body { id, s } => {
                    // Non-static lifetimes are prohibited in anonymous constants without
                    // `generic_const_exprs`.
                    self.maybe_emit_forbidden_non_static_lifetime_error(id, lifetime_ref);

                    outermost_body = Some(id);
                    scope = s;
                }

                Scope::Root => {
                    break None;
                }

                Scope::Binder { ref lifetimes, scope_type, s, .. } => {
                    match lifetime_ref.name {
                        LifetimeName::Param(param_name) => {
                            if let Some(&def) = lifetimes.get(&param_name.normalize_to_macros_2_0())
                            {
                                break Some(def.shifted(late_depth));
                            }
                        }
                        _ => bug!("expected LifetimeName::Param"),
                    }
                    match scope_type {
                        BinderScopeType::Normal => late_depth += 1,
                        BinderScopeType::Concatenating => {}
                    }
                    scope = s;
                }

                Scope::Elision { s, .. }
                | Scope::ObjectLifetimeDefault { s, .. }
                | Scope::Supertrait { s, .. }
                | Scope::TraitRefBoundary { s, .. } => {
                    scope = s;
                }
            }
        };

        if let Some(mut def) = result {
            if let Region::EarlyBound(..) = def {
                // Do not free early-bound regions, only late-bound ones.
            } else if let Some(body_id) = outermost_body {
                let fn_id = self.tcx.hir().body_owner(body_id);
                match self.tcx.hir().get(fn_id) {
                    Node::Item(&hir::Item { kind: hir::ItemKind::Fn(..), .. })
                    | Node::TraitItem(&hir::TraitItem {
                        kind: hir::TraitItemKind::Fn(..), ..
                    })
                    | Node::ImplItem(&hir::ImplItem { kind: hir::ImplItemKind::Fn(..), .. }) => {
                        let scope = self.tcx.hir().local_def_id(fn_id);
                        def = Region::Free(scope.to_def_id(), def.id().unwrap());
                    }
                    _ => {}
                }
            }

            // Check for fn-syntax conflicts with in-band lifetime definitions
            if !self.trait_definition_only && self.is_in_fn_syntax {
                match def {
                    Region::EarlyBound(_, _, LifetimeDefOrigin::InBand)
                    | Region::LateBound(_, _, _, LifetimeDefOrigin::InBand) => {
                        struct_span_err!(
                            self.tcx.sess,
                            lifetime_ref.span,
                            E0687,
                            "lifetimes used in `fn` or `Fn` syntax must be \
                             explicitly declared using `<...>` binders"
                        )
                        .span_label(lifetime_ref.span, "in-band lifetime definition")
                        .emit();
                    }

                    Region::Static
                    | Region::EarlyBound(
                        _,
                        _,
                        LifetimeDefOrigin::ExplicitOrElided | LifetimeDefOrigin::Error,
                    )
                    | Region::LateBound(
                        _,
                        _,
                        _,
                        LifetimeDefOrigin::ExplicitOrElided | LifetimeDefOrigin::Error,
                    )
                    | Region::LateBoundAnon(..)
                    | Region::Free(..) => {}
                }
            }

            self.insert_lifetime(lifetime_ref, def);
        } else {
            self.emit_undeclared_lifetime_error(lifetime_ref);
        }
    }

    fn visit_segment_args(
        &mut self,
        res: Res,
        depth: usize,
        generic_args: &'tcx hir::GenericArgs<'tcx>,
    ) {
        debug!(
            "visit_segment_args(res={:?}, depth={:?}, generic_args={:?})",
            res, depth, generic_args,
        );

        if generic_args.parenthesized {
            let was_in_fn_syntax = self.is_in_fn_syntax;
            self.is_in_fn_syntax = true;
            self.visit_fn_like_elision(generic_args.inputs(), Some(generic_args.bindings[0].ty()));
            self.is_in_fn_syntax = was_in_fn_syntax;
            return;
        }

        let mut elide_lifetimes = true;
        let lifetimes: Vec<_> = generic_args
            .args
            .iter()
            .filter_map(|arg| match arg {
                hir::GenericArg::Lifetime(lt) => {
                    if !lt.is_elided() {
                        elide_lifetimes = false;
                    }
                    Some(lt)
                }
                _ => None,
            })
            .collect();
        // We short-circuit here if all are elided in order to pluralize
        // possible errors
        if elide_lifetimes {
            self.resolve_elided_lifetimes(&lifetimes);
        } else {
            lifetimes.iter().for_each(|lt| self.visit_lifetime(lt));
        }

        // Figure out if this is a type/trait segment,
        // which requires object lifetime defaults.
        let parent_def_id = |this: &mut Self, def_id: DefId| {
            let def_key = this.tcx.def_key(def_id);
            DefId { krate: def_id.krate, index: def_key.parent.expect("missing parent") }
        };
        let type_def_id = match res {
            Res::Def(DefKind::AssocTy, def_id) if depth == 1 => Some(parent_def_id(self, def_id)),
            Res::Def(DefKind::Variant, def_id) if depth == 0 => Some(parent_def_id(self, def_id)),
            Res::Def(
                DefKind::Struct
                | DefKind::Union
                | DefKind::Enum
                | DefKind::TyAlias
                | DefKind::Trait,
                def_id,
            ) if depth == 0 => Some(def_id),
            _ => None,
        };

        debug!("visit_segment_args: type_def_id={:?}", type_def_id);

        // Compute a vector of defaults, one for each type parameter,
        // per the rules given in RFCs 599 and 1156. Example:
        //
        // ```rust
        // struct Foo<'a, T: 'a, U> { }
        // ```
        //
        // If you have `Foo<'x, dyn Bar, dyn Baz>`, we want to default
        // `dyn Bar` to `dyn Bar + 'x` (because of the `T: 'a` bound)
        // and `dyn Baz` to `dyn Baz + 'static` (because there is no
        // such bound).
        //
        // Therefore, we would compute `object_lifetime_defaults` to a
        // vector like `['x, 'static]`. Note that the vector only
        // includes type parameters.
        let object_lifetime_defaults = type_def_id.map_or_else(Vec::new, |def_id| {
            let in_body = {
                let mut scope = self.scope;
                loop {
                    match *scope {
                        Scope::Root => break false,

                        Scope::Body { .. } => break true,

                        Scope::Binder { s, .. }
                        | Scope::Elision { s, .. }
                        | Scope::ObjectLifetimeDefault { s, .. }
                        | Scope::Supertrait { s, .. }
                        | Scope::TraitRefBoundary { s, .. } => {
                            scope = s;
                        }
                    }
                }
            };

            let map = &self.map;
            let set_to_region = |set: &ObjectLifetimeDefault| match *set {
                Set1::Empty => {
                    if in_body {
                        None
                    } else {
                        Some(Region::Static)
                    }
                }
                Set1::One(r) => {
                    let lifetimes = generic_args.args.iter().filter_map(|arg| match arg {
                        GenericArg::Lifetime(lt) => Some(lt),
                        _ => None,
                    });
                    r.subst(lifetimes, map)
                }
                Set1::Many => None,
            };
            if let Some(def_id) = def_id.as_local() {
                let id = self.tcx.hir().local_def_id_to_hir_id(def_id);
                self.tcx.object_lifetime_defaults(id).unwrap().iter().map(set_to_region).collect()
            } else {
                let tcx = self.tcx;
                self.xcrate_object_lifetime_defaults
                    .entry(def_id)
                    .or_insert_with(|| {
                        tcx.generics_of(def_id)
                            .params
                            .iter()
                            .filter_map(|param| match param.kind {
                                GenericParamDefKind::Type { object_lifetime_default, .. } => {
                                    Some(object_lifetime_default)
                                }
                                GenericParamDefKind::Lifetime
                                | GenericParamDefKind::Const { .. } => None,
                            })
                            .collect()
                    })
                    .iter()
                    .map(set_to_region)
                    .collect()
            }
        });

        debug!("visit_segment_args: object_lifetime_defaults={:?}", object_lifetime_defaults);

        let mut i = 0;
        for arg in generic_args.args {
            match arg {
                GenericArg::Lifetime(_) => {}
                GenericArg::Type(ty) => {
                    if let Some(&lt) = object_lifetime_defaults.get(i) {
                        let scope = Scope::ObjectLifetimeDefault { lifetime: lt, s: self.scope };
                        self.with(scope, |_, this| this.visit_ty(ty));
                    } else {
                        self.visit_ty(ty);
                    }
                    i += 1;
                }
                GenericArg::Const(ct) => {
                    self.visit_anon_const(&ct.value);
                }
                GenericArg::Infer(inf) => {
                    self.visit_id(inf.hir_id);
                    if inf.kind.is_type() {
                        i += 1;
                    }
                }
            }
        }

        // Hack: when resolving the type `XX` in binding like `dyn
        // Foo<'b, Item = XX>`, the current object-lifetime default
        // would be to examine the trait `Foo` to check whether it has
        // a lifetime bound declared on `Item`. e.g., if `Foo` is
        // declared like so, then the default object lifetime bound in
        // `XX` should be `'b`:
        //
        // ```rust
        // trait Foo<'a> {
        //   type Item: 'a;
        // }
        // ```
        //
        // but if we just have `type Item;`, then it would be
        // `'static`. However, we don't get all of this logic correct.
        //
        // Instead, we do something hacky: if there are no lifetime parameters
        // to the trait, then we simply use a default object lifetime
        // bound of `'static`, because there is no other possibility. On the other hand,
        // if there ARE lifetime parameters, then we require the user to give an
        // explicit bound for now.
        //
        // This is intended to leave room for us to implement the
        // correct behavior in the future.
        let has_lifetime_parameter =
            generic_args.args.iter().any(|arg| matches!(arg, GenericArg::Lifetime(_)));

        // Resolve lifetimes found in the bindings, so either in the type `XX` in `Item = XX` or
        // in the trait ref `YY<...>` in `Item: YY<...>`.
        for binding in generic_args.bindings {
            let scope = Scope::ObjectLifetimeDefault {
                lifetime: if has_lifetime_parameter { None } else { Some(Region::Static) },
                s: self.scope,
            };
            if let Some(type_def_id) = type_def_id {
                let lifetimes = LifetimeContext::supertrait_hrtb_lifetimes(
                    self.tcx,
                    type_def_id,
                    binding.ident,
                );
                self.with(scope, |_, this| {
                    let scope = Scope::Supertrait {
                        lifetimes: lifetimes.unwrap_or_default(),
                        s: this.scope,
                    };
                    this.with(scope, |_, this| this.visit_assoc_type_binding(binding));
                });
            } else {
                self.with(scope, |_, this| this.visit_assoc_type_binding(binding));
            }
        }
    }

    /// Returns all the late-bound vars that come into scope from supertrait HRTBs, based on the
    /// associated type name and starting trait.
    /// For example, imagine we have
    /// ```rust
    /// trait Foo<'a, 'b> {
    ///   type As;
    /// }
    /// trait Bar<'b>: for<'a> Foo<'a, 'b> {}
    /// trait Bar: for<'b> Bar<'b> {}
    /// ```
    /// In this case, if we wanted to the supertrait HRTB lifetimes for `As` on
    /// the starting trait `Bar`, we would return `Some(['b, 'a])`.
    fn supertrait_hrtb_lifetimes(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        assoc_name: Ident,
    ) -> Option<Vec<ty::BoundVariableKind>> {
        let trait_defines_associated_type_named = |trait_def_id: DefId| {
            tcx.associated_items(trait_def_id)
                .find_by_name_and_kind(tcx, assoc_name, ty::AssocKind::Type, trait_def_id)
                .is_some()
        };

        use smallvec::{smallvec, SmallVec};
        let mut stack: SmallVec<[(DefId, SmallVec<[ty::BoundVariableKind; 8]>); 8]> =
            smallvec![(def_id, smallvec![])];
        let mut visited: FxHashSet<DefId> = FxHashSet::default();
        loop {
            let (def_id, bound_vars) = match stack.pop() {
                Some(next) => next,
                None => break None,
            };
            // See issue #83753. If someone writes an associated type on a non-trait, just treat it as
            // there being no supertrait HRTBs.
            match tcx.def_kind(def_id) {
                DefKind::Trait | DefKind::TraitAlias | DefKind::Impl => {}
                _ => break None,
            }

            if trait_defines_associated_type_named(def_id) {
                break Some(bound_vars.into_iter().collect());
            }
            let predicates =
                tcx.super_predicates_that_define_assoc_type((def_id, Some(assoc_name)));
            let obligations = predicates.predicates.iter().filter_map(|&(pred, _)| {
                let bound_predicate = pred.kind();
                match bound_predicate.skip_binder() {
                    ty::PredicateKind::Trait(data) => {
                        // The order here needs to match what we would get from `subst_supertrait`
                        let pred_bound_vars = bound_predicate.bound_vars();
                        let mut all_bound_vars = bound_vars.clone();
                        all_bound_vars.extend(pred_bound_vars.iter());
                        let super_def_id = data.trait_ref.def_id;
                        Some((super_def_id, all_bound_vars))
                    }
                    _ => None,
                }
            });

            let obligations = obligations.filter(|o| visited.insert(o.0));
            stack.extend(obligations);
        }
    }

    #[tracing::instrument(level = "debug", skip(self))]
    fn visit_fn_like_elision(
        &mut self,
        inputs: &'tcx [hir::Ty<'tcx>],
        output: Option<&'tcx hir::Ty<'tcx>>,
    ) {
        debug!("visit_fn_like_elision: enter");
        let mut scope = &*self.scope;
        let hir_id = loop {
            match scope {
                Scope::Binder { hir_id, .. } => {
                    break *hir_id;
                }
                Scope::ObjectLifetimeDefault { ref s, .. }
                | Scope::Elision { ref s, .. }
                | Scope::Supertrait { ref s, .. }
                | Scope::TraitRefBoundary { ref s, .. } => {
                    scope = *s;
                }
                Scope::Root | Scope::Body { .. } => {
                    // See issues #83907 and #83693. Just bail out from looking inside.
                    self.tcx.sess.delay_span_bug(
                        rustc_span::DUMMY_SP,
                        "In fn_like_elision without appropriate scope above",
                    );
                    return;
                }
            }
        };
        // While not strictly necessary, we gather anon lifetimes *before* actually
        // visiting the argument types.
        let mut gather = GatherAnonLifetimes { anon_count: 0 };
        for input in inputs {
            gather.visit_ty(input);
        }
        let late_bound_vars = self.map.late_bound_vars.entry(hir_id).or_default();
        let named_late_bound_vars = late_bound_vars.len() as u32;
        late_bound_vars.extend(
            (0..gather.anon_count).map(|var| ty::BoundVariableKind::Region(ty::BrAnon(var))),
        );
        let arg_scope = Scope::Elision {
            elide: Elide::FreshLateAnon(named_late_bound_vars, Cell::new(0)),
            s: self.scope,
        };
        self.with(arg_scope, |_, this| {
            for input in inputs {
                this.visit_ty(input);
            }
        });

        let output = match output {
            Some(ty) => ty,
            None => return,
        };

        debug!("determine output");

        // Figure out if there's a body we can get argument names from,
        // and whether there's a `self` argument (treated specially).
        let mut assoc_item_kind = None;
        let mut impl_self = None;
        let parent = self.tcx.hir().get_parent_node(output.hir_id);
        let body = match self.tcx.hir().get(parent) {
            // `fn` definitions and methods.
            Node::Item(&hir::Item { kind: hir::ItemKind::Fn(.., body), .. }) => Some(body),

            Node::TraitItem(&hir::TraitItem { kind: hir::TraitItemKind::Fn(_, ref m), .. }) => {
                if let hir::ItemKind::Trait(.., ref trait_items) =
                    self.tcx.hir().expect_item(self.tcx.hir().get_parent_item(parent)).kind
                {
                    assoc_item_kind =
                        trait_items.iter().find(|ti| ti.id.hir_id() == parent).map(|ti| ti.kind);
                }
                match *m {
                    hir::TraitFn::Required(_) => None,
                    hir::TraitFn::Provided(body) => Some(body),
                }
            }

            Node::ImplItem(&hir::ImplItem { kind: hir::ImplItemKind::Fn(_, body), .. }) => {
                if let hir::ItemKind::Impl(hir::Impl { ref self_ty, ref items, .. }) =
                    self.tcx.hir().expect_item(self.tcx.hir().get_parent_item(parent)).kind
                {
                    impl_self = Some(self_ty);
                    assoc_item_kind =
                        items.iter().find(|ii| ii.id.hir_id() == parent).map(|ii| ii.kind);
                }
                Some(body)
            }

            // Foreign functions, `fn(...) -> R` and `Trait(...) -> R` (both types and bounds).
            Node::ForeignItem(_) | Node::Ty(_) | Node::TraitRef(_) => None,
            // Everything else (only closures?) doesn't
            // actually enjoy elision in return types.
            _ => {
                self.visit_ty(output);
                return;
            }
        };

        let has_self = match assoc_item_kind {
            Some(hir::AssocItemKind::Fn { has_self }) => has_self,
            _ => false,
        };

        // In accordance with the rules for lifetime elision, we can determine
        // what region to use for elision in the output type in two ways.
        // First (determined here), if `self` is by-reference, then the
        // implied output region is the region of the self parameter.
        if has_self {
            struct SelfVisitor<'a> {
                map: &'a NamedRegionMap,
                impl_self: Option<&'a hir::TyKind<'a>>,
                lifetime: Set1<Region>,
            }

            impl SelfVisitor<'_> {
                // Look for `self: &'a Self` - also desugared from `&'a self`,
                // and if that matches, use it for elision and return early.
                fn is_self_ty(&self, res: Res) -> bool {
                    if let Res::SelfTy(..) = res {
                        return true;
                    }

                    // Can't always rely on literal (or implied) `Self` due
                    // to the way elision rules were originally specified.
                    if let Some(&hir::TyKind::Path(hir::QPath::Resolved(None, ref path))) =
                        self.impl_self
                    {
                        match path.res {
                            // Permit the types that unambiguously always
                            // result in the same type constructor being used
                            // (it can't differ between `Self` and `self`).
                            Res::Def(DefKind::Struct | DefKind::Union | DefKind::Enum, _)
                            | Res::PrimTy(_) => return res == path.res,
                            _ => {}
                        }
                    }

                    false
                }
            }

            impl<'a> Visitor<'a> for SelfVisitor<'a> {
                type Map = intravisit::ErasedMap<'a>;

                fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
                    NestedVisitorMap::None
                }

                fn visit_ty(&mut self, ty: &'a hir::Ty<'a>) {
                    if let hir::TyKind::Rptr(lifetime_ref, ref mt) = ty.kind {
                        if let hir::TyKind::Path(hir::QPath::Resolved(None, ref path)) = mt.ty.kind
                        {
                            if self.is_self_ty(path.res) {
                                if let Some(lifetime) = self.map.defs.get(&lifetime_ref.hir_id) {
                                    self.lifetime.insert(*lifetime);
                                }
                            }
                        }
                    }
                    intravisit::walk_ty(self, ty)
                }
            }

            let mut visitor = SelfVisitor {
                map: self.map,
                impl_self: impl_self.map(|ty| &ty.kind),
                lifetime: Set1::Empty,
            };
            visitor.visit_ty(&inputs[0]);
            if let Set1::One(lifetime) = visitor.lifetime {
                let scope = Scope::Elision { elide: Elide::Exact(lifetime), s: self.scope };
                self.with(scope, |_, this| this.visit_ty(output));
                return;
            }
        }

        // Second, if there was exactly one lifetime (either a substitution or a
        // reference) in the arguments, then any anonymous regions in the output
        // have that lifetime.
        let mut possible_implied_output_region = None;
        let mut lifetime_count = 0;
        let arg_lifetimes = inputs
            .iter()
            .enumerate()
            .skip(has_self as usize)
            .map(|(i, input)| {
                let mut gather = GatherLifetimes {
                    map: self.map,
                    outer_index: ty::INNERMOST,
                    have_bound_regions: false,
                    lifetimes: Default::default(),
                };
                gather.visit_ty(input);

                lifetime_count += gather.lifetimes.len();

                if lifetime_count == 1 && gather.lifetimes.len() == 1 {
                    // there's a chance that the unique lifetime of this
                    // iteration will be the appropriate lifetime for output
                    // parameters, so lets store it.
                    possible_implied_output_region = gather.lifetimes.iter().cloned().next();
                }

                ElisionFailureInfo {
                    parent: body,
                    index: i,
                    lifetime_count: gather.lifetimes.len(),
                    have_bound_regions: gather.have_bound_regions,
                    span: input.span,
                }
            })
            .collect();

        let elide = if lifetime_count == 1 {
            Elide::Exact(possible_implied_output_region.unwrap())
        } else {
            Elide::Error(arg_lifetimes)
        };

        debug!(?elide);

        let scope = Scope::Elision { elide, s: self.scope };
        self.with(scope, |_, this| this.visit_ty(output));

        struct GatherLifetimes<'a> {
            map: &'a NamedRegionMap,
            outer_index: ty::DebruijnIndex,
            have_bound_regions: bool,
            lifetimes: FxHashSet<Region>,
        }

        impl<'v, 'a> Visitor<'v> for GatherLifetimes<'a> {
            type Map = intravisit::ErasedMap<'v>;

            fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
                NestedVisitorMap::None
            }

            fn visit_ty(&mut self, ty: &hir::Ty<'_>) {
                if let hir::TyKind::BareFn(_) = ty.kind {
                    self.outer_index.shift_in(1);
                }
                match ty.kind {
                    hir::TyKind::TraitObject(bounds, ref lifetime, _) => {
                        for bound in bounds {
                            self.visit_poly_trait_ref(bound, hir::TraitBoundModifier::None);
                        }

                        // Stay on the safe side and don't include the object
                        // lifetime default (which may not end up being used).
                        if !lifetime.is_elided() {
                            self.visit_lifetime(lifetime);
                        }
                    }
                    _ => {
                        intravisit::walk_ty(self, ty);
                    }
                }
                if let hir::TyKind::BareFn(_) = ty.kind {
                    self.outer_index.shift_out(1);
                }
            }

            fn visit_generic_param(&mut self, param: &hir::GenericParam<'_>) {
                if let hir::GenericParamKind::Lifetime { .. } = param.kind {
                    // FIXME(eddyb) Do we want this? It only makes a difference
                    // if this `for<'a>` lifetime parameter is never used.
                    self.have_bound_regions = true;
                }

                intravisit::walk_generic_param(self, param);
            }

            fn visit_poly_trait_ref(
                &mut self,
                trait_ref: &hir::PolyTraitRef<'_>,
                modifier: hir::TraitBoundModifier,
            ) {
                self.outer_index.shift_in(1);
                intravisit::walk_poly_trait_ref(self, trait_ref, modifier);
                self.outer_index.shift_out(1);
            }

            fn visit_param_bound(&mut self, bound: &hir::GenericBound<'_>) {
                if let hir::GenericBound::LangItemTrait { .. } = bound {
                    self.outer_index.shift_in(1);
                    intravisit::walk_param_bound(self, bound);
                    self.outer_index.shift_out(1);
                } else {
                    intravisit::walk_param_bound(self, bound);
                }
            }

            fn visit_lifetime(&mut self, lifetime_ref: &hir::Lifetime) {
                if let Some(&lifetime) = self.map.defs.get(&lifetime_ref.hir_id) {
                    match lifetime {
                        Region::LateBound(debruijn, _, _, _)
                        | Region::LateBoundAnon(debruijn, _, _)
                            if debruijn < self.outer_index =>
                        {
                            self.have_bound_regions = true;
                        }
                        _ => {
                            // FIXME(jackh726): nested trait refs?
                            self.lifetimes.insert(lifetime.shifted_out_to_binder(self.outer_index));
                        }
                    }
                }
            }
        }

        struct GatherAnonLifetimes {
            anon_count: u32,
        }
        impl<'v> Visitor<'v> for GatherAnonLifetimes {
            type Map = intravisit::ErasedMap<'v>;

            fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
                NestedVisitorMap::None
            }

            fn visit_ty(&mut self, ty: &hir::Ty<'_>) {
                // If we enter a `BareFn`, then we enter a *new* binding scope
                if let hir::TyKind::BareFn(_) = ty.kind {
                    return;
                }
                intravisit::walk_ty(self, ty);
            }

            fn visit_generic_args(
                &mut self,
                path_span: Span,
                generic_args: &'v hir::GenericArgs<'v>,
            ) {
                // parenthesized args enter a new elison scope
                if generic_args.parenthesized {
                    return;
                }
                intravisit::walk_generic_args(self, path_span, generic_args)
            }

            fn visit_lifetime(&mut self, lifetime_ref: &hir::Lifetime) {
                if lifetime_ref.is_elided() {
                    self.anon_count += 1;
                }
            }
        }
    }

    fn resolve_elided_lifetimes(&mut self, lifetime_refs: &[&'tcx hir::Lifetime]) {
        debug!("resolve_elided_lifetimes(lifetime_refs={:?})", lifetime_refs);

        if lifetime_refs.is_empty() {
            return;
        }

        let mut late_depth = 0;
        let mut scope = self.scope;
        let mut lifetime_names = FxHashSet::default();
        let mut lifetime_spans = vec![];
        let error = loop {
            match *scope {
                // Do not assign any resolution, it will be inferred.
                Scope::Body { .. } => return,

                Scope::Root => break None,

                Scope::Binder { s, ref lifetimes, scope_type, .. } => {
                    // collect named lifetimes for suggestions
                    for name in lifetimes.keys() {
                        if let hir::ParamName::Plain(name) = name {
                            lifetime_names.insert(name.name);
                            lifetime_spans.push(name.span);
                        }
                    }
                    match scope_type {
                        BinderScopeType::Normal => late_depth += 1,
                        BinderScopeType::Concatenating => {}
                    }
                    scope = s;
                }

                Scope::Elision { ref elide, ref s, .. } => {
                    let lifetime = match *elide {
                        Elide::FreshLateAnon(named_late_bound_vars, ref counter) => {
                            for lifetime_ref in lifetime_refs {
                                let lifetime = Region::late_anon(named_late_bound_vars, counter)
                                    .shifted(late_depth);

                                self.insert_lifetime(lifetime_ref, lifetime);
                            }
                            return;
                        }
                        Elide::Exact(l) => l.shifted(late_depth),
                        Elide::Error(ref e) => {
                            let mut scope = s;
                            loop {
                                match scope {
                                    Scope::Binder { ref lifetimes, s, .. } => {
                                        // Collect named lifetimes for suggestions.
                                        for name in lifetimes.keys() {
                                            if let hir::ParamName::Plain(name) = name {
                                                lifetime_names.insert(name.name);
                                                lifetime_spans.push(name.span);
                                            }
                                        }
                                        scope = s;
                                    }
                                    Scope::ObjectLifetimeDefault { ref s, .. }
                                    | Scope::Elision { ref s, .. }
                                    | Scope::TraitRefBoundary { ref s, .. } => {
                                        scope = s;
                                    }
                                    _ => break,
                                }
                            }
                            break Some(&e[..]);
                        }
                        Elide::Forbid => break None,
                    };
                    for lifetime_ref in lifetime_refs {
                        self.insert_lifetime(lifetime_ref, lifetime);
                    }
                    return;
                }

                Scope::ObjectLifetimeDefault { s, .. }
                | Scope::Supertrait { s, .. }
                | Scope::TraitRefBoundary { s, .. } => {
                    scope = s;
                }
            }
        };

        // If we specifically need the `scope_for_path` map, then we're in the
        // diagnostic pass and we don't want to emit more errors.
        if self.map.scope_for_path.is_some() {
            self.tcx.sess.delay_span_bug(
                rustc_span::DUMMY_SP,
                "Encountered unexpected errors during diagnostics related part",
            );
            return;
        }

        let mut spans: Vec<_> = lifetime_refs.iter().map(|lt| lt.span).collect();
        spans.sort();
        let mut spans_dedup = spans.clone();
        spans_dedup.dedup();
        let spans_with_counts: Vec<_> = spans_dedup
            .into_iter()
            .map(|sp| (sp, spans.iter().filter(|nsp| *nsp == &sp).count()))
            .collect();

        let mut err = self.report_missing_lifetime_specifiers(spans.clone(), lifetime_refs.len());

        if let Some(params) = error {
            // If there's no lifetime available, suggest `'static`.
            if self.report_elision_failure(&mut err, params) && lifetime_names.is_empty() {
                lifetime_names.insert(kw::StaticLifetime);
            }
        }

        self.add_missing_lifetime_specifiers_label(
            &mut err,
            spans_with_counts,
            &lifetime_names,
            lifetime_spans,
            error.unwrap_or(&[]),
        );
        err.emit();
    }

    fn report_elision_failure(
        &mut self,
        db: &mut DiagnosticBuilder<'_>,
        params: &[ElisionFailureInfo],
    ) -> bool /* add `'static` lifetime to lifetime list */ {
        let mut m = String::new();
        let len = params.len();

        let elided_params: Vec<_> =
            params.iter().cloned().filter(|info| info.lifetime_count > 0).collect();

        let elided_len = elided_params.len();

        for (i, info) in elided_params.into_iter().enumerate() {
            let ElisionFailureInfo { parent, index, lifetime_count: n, have_bound_regions, span } =
                info;

            db.span_label(span, "");
            let help_name = if let Some(ident) =
                parent.and_then(|body| self.tcx.hir().body(body).params[index].pat.simple_ident())
            {
                format!("`{}`", ident)
            } else {
                format!("argument {}", index + 1)
            };

            m.push_str(
                &(if n == 1 {
                    help_name
                } else {
                    format!(
                        "one of {}'s {} {}lifetimes",
                        help_name,
                        n,
                        if have_bound_regions { "free " } else { "" }
                    )
                })[..],
            );

            if elided_len == 2 && i == 0 {
                m.push_str(" or ");
            } else if i + 2 == elided_len {
                m.push_str(", or ");
            } else if i != elided_len - 1 {
                m.push_str(", ");
            }
        }

        if len == 0 {
            db.help(
                "this function's return type contains a borrowed value, \
                 but there is no value for it to be borrowed from",
            );
            true
        } else if elided_len == 0 {
            db.help(
                "this function's return type contains a borrowed value with \
                 an elided lifetime, but the lifetime cannot be derived from \
                 the arguments",
            );
            true
        } else if elided_len == 1 {
            db.help(&format!(
                "this function's return type contains a borrowed value, \
                 but the signature does not say which {} it is borrowed from",
                m
            ));
            false
        } else {
            db.help(&format!(
                "this function's return type contains a borrowed value, \
                 but the signature does not say whether it is borrowed from {}",
                m
            ));
            false
        }
    }

    fn resolve_object_lifetime_default(&mut self, lifetime_ref: &'tcx hir::Lifetime) {
        debug!("resolve_object_lifetime_default(lifetime_ref={:?})", lifetime_ref);
        let mut late_depth = 0;
        let mut scope = self.scope;
        let lifetime = loop {
            match *scope {
                Scope::Binder { s, scope_type, .. } => {
                    match scope_type {
                        BinderScopeType::Normal => late_depth += 1,
                        BinderScopeType::Concatenating => {}
                    }
                    scope = s;
                }

                Scope::Root | Scope::Elision { .. } => break Region::Static,

                Scope::Body { .. } | Scope::ObjectLifetimeDefault { lifetime: None, .. } => return,

                Scope::ObjectLifetimeDefault { lifetime: Some(l), .. } => break l,

                Scope::Supertrait { s, .. } | Scope::TraitRefBoundary { s, .. } => {
                    scope = s;
                }
            }
        };
        self.insert_lifetime(lifetime_ref, lifetime.shifted(late_depth));
    }

    fn check_lifetime_params(
        &mut self,
        old_scope: ScopeRef<'_>,
        params: &'tcx [hir::GenericParam<'tcx>],
    ) {
        let lifetimes: Vec<_> = params
            .iter()
            .filter_map(|param| match param.kind {
                GenericParamKind::Lifetime { .. } => {
                    Some((param, param.name.normalize_to_macros_2_0()))
                }
                _ => None,
            })
            .collect();
        for (i, (lifetime_i, lifetime_i_name)) in lifetimes.iter().enumerate() {
            if let hir::ParamName::Plain(_) = lifetime_i_name {
                let name = lifetime_i_name.ident().name;
                if name == kw::UnderscoreLifetime || name == kw::StaticLifetime {
                    let mut err = struct_span_err!(
                        self.tcx.sess,
                        lifetime_i.span,
                        E0262,
                        "invalid lifetime parameter name: `{}`",
                        lifetime_i.name.ident(),
                    );
                    err.span_label(
                        lifetime_i.span,
                        format!("{} is a reserved lifetime name", name),
                    );
                    err.emit();
                }
            }

            // It is a hard error to shadow a lifetime within the same scope.
            for (lifetime_j, lifetime_j_name) in lifetimes.iter().skip(i + 1) {
                if lifetime_i_name == lifetime_j_name {
                    struct_span_err!(
                        self.tcx.sess,
                        lifetime_j.span,
                        E0263,
                        "lifetime name `{}` declared twice in the same scope",
                        lifetime_j.name.ident()
                    )
                    .span_label(lifetime_j.span, "declared twice")
                    .span_label(lifetime_i.span, "previous declaration here")
                    .emit();
                }
            }

            // It is a soft error to shadow a lifetime within a parent scope.
            self.check_lifetime_param_for_shadowing(old_scope, &lifetime_i);

            for bound in lifetime_i.bounds {
                match bound {
                    hir::GenericBound::Outlives(ref lt) => match lt.name {
                        hir::LifetimeName::Underscore => self.tcx.sess.delay_span_bug(
                            lt.span,
                            "use of `'_` in illegal place, but not caught by lowering",
                        ),
                        hir::LifetimeName::Static => {
                            self.insert_lifetime(lt, Region::Static);
                            self.tcx
                                .sess
                                .struct_span_warn(
                                    lifetime_i.span.to(lt.span),
                                    &format!(
                                        "unnecessary lifetime parameter `{}`",
                                        lifetime_i.name.ident(),
                                    ),
                                )
                                .help(&format!(
                                    "you can use the `'static` lifetime directly, in place of `{}`",
                                    lifetime_i.name.ident(),
                                ))
                                .emit();
                        }
                        hir::LifetimeName::Param(_) | hir::LifetimeName::Implicit => {
                            self.resolve_lifetime_ref(lt);
                        }
                        hir::LifetimeName::ImplicitObjectLifetimeDefault => {
                            self.tcx.sess.delay_span_bug(
                                lt.span,
                                "lowering generated `ImplicitObjectLifetimeDefault` \
                                 outside of an object type",
                            )
                        }
                        hir::LifetimeName::Error => {
                            // No need to do anything, error already reported.
                        }
                    },
                    _ => bug!(),
                }
            }
        }
    }

    fn check_lifetime_param_for_shadowing(
        &self,
        mut old_scope: ScopeRef<'_>,
        param: &'tcx hir::GenericParam<'tcx>,
    ) {
        for label in &self.labels_in_fn {
            // FIXME (#24278): non-hygienic comparison
            if param.name.ident().name == label.name {
                signal_shadowing_problem(
                    self.tcx,
                    label.name,
                    original_label(label.span),
                    shadower_lifetime(&param),
                );
                return;
            }
        }

        loop {
            match *old_scope {
                Scope::Body { s, .. }
                | Scope::Elision { s, .. }
                | Scope::ObjectLifetimeDefault { s, .. }
                | Scope::Supertrait { s, .. }
                | Scope::TraitRefBoundary { s, .. } => {
                    old_scope = s;
                }

                Scope::Root => {
                    return;
                }

                Scope::Binder { ref lifetimes, s, .. } => {
                    if let Some(&def) = lifetimes.get(&param.name.normalize_to_macros_2_0()) {
                        let hir_id =
                            self.tcx.hir().local_def_id_to_hir_id(def.id().unwrap().expect_local());

                        signal_shadowing_problem(
                            self.tcx,
                            param.name.ident().name,
                            original_lifetime(self.tcx.hir().span(hir_id)),
                            shadower_lifetime(&param),
                        );
                        return;
                    }

                    old_scope = s;
                }
            }
        }
    }

    /// Returns `true` if, in the current scope, replacing `'_` would be
    /// equivalent to a single-use lifetime.
    fn track_lifetime_uses(&self) -> bool {
        let mut scope = self.scope;
        loop {
            match *scope {
                Scope::Root => break false,

                // Inside of items, it depends on the kind of item.
                Scope::Binder { track_lifetime_uses, .. } => break track_lifetime_uses,

                // Inside a body, `'_` will use an inference variable,
                // should be fine.
                Scope::Body { .. } => break true,

                // A lifetime only used in a fn argument could as well
                // be replaced with `'_`, as that would generate a
                // fresh name, too.
                Scope::Elision { elide: Elide::FreshLateAnon(..), .. } => break true,

                // In the return type or other such place, `'_` is not
                // going to make a fresh name, so we cannot
                // necessarily replace a single-use lifetime with
                // `'_`.
                Scope::Elision {
                    elide: Elide::Exact(_) | Elide::Error(_) | Elide::Forbid, ..
                } => break false,

                Scope::ObjectLifetimeDefault { s, .. }
                | Scope::Supertrait { s, .. }
                | Scope::TraitRefBoundary { s, .. } => scope = s,
            }
        }
    }

    #[tracing::instrument(level = "debug", skip(self))]
    fn insert_lifetime(&mut self, lifetime_ref: &'tcx hir::Lifetime, def: Region) {
        debug!(
            node = ?self.tcx.hir().node_to_string(lifetime_ref.hir_id),
            span = ?self.tcx.sess.source_map().span_to_diagnostic_string(lifetime_ref.span)
        );
        self.map.defs.insert(lifetime_ref.hir_id, def);

        match def {
            Region::LateBoundAnon(..) | Region::Static => {
                // These are anonymous lifetimes or lifetimes that are not declared.
            }

            Region::Free(_, def_id)
            | Region::LateBound(_, _, def_id, _)
            | Region::EarlyBound(_, def_id, _) => {
                // A lifetime declared by the user.
                let track_lifetime_uses = self.track_lifetime_uses();
                debug!(?track_lifetime_uses);
                if track_lifetime_uses && !self.lifetime_uses.contains_key(&def_id) {
                    debug!("first use of {:?}", def_id);
                    self.lifetime_uses.insert(def_id, LifetimeUseSet::One(lifetime_ref));
                } else {
                    debug!("many uses of {:?}", def_id);
                    self.lifetime_uses.insert(def_id, LifetimeUseSet::Many);
                }
            }
        }
    }

    /// Sometimes we resolve a lifetime, but later find that it is an
    /// error (esp. around impl trait). In that case, we remove the
    /// entry into `map.defs` so as not to confuse later code.
    fn uninsert_lifetime_on_error(&mut self, lifetime_ref: &'tcx hir::Lifetime, bad_def: Region) {
        let old_value = self.map.defs.remove(&lifetime_ref.hir_id);
        assert_eq!(old_value, Some(bad_def));
    }
}

/// Detects late-bound lifetimes and inserts them into
/// `map.late_bound`.
///
/// A region declared on a fn is **late-bound** if:
/// - it is constrained by an argument type;
/// - it does not appear in a where-clause.
///
/// "Constrained" basically means that it appears in any type but
/// not amongst the inputs to a projection. In other words, `<&'a
/// T as Trait<''b>>::Foo` does not constrain `'a` or `'b`.
#[tracing::instrument(level = "debug", skip(map))]
fn insert_late_bound_lifetimes(
    map: &mut NamedRegionMap,
    decl: &hir::FnDecl<'_>,
    generics: &hir::Generics<'_>,
) {
    let mut constrained_by_input = ConstrainedCollector::default();
    for arg_ty in decl.inputs {
        constrained_by_input.visit_ty(arg_ty);
    }

    let mut appears_in_output = AllCollector::default();
    intravisit::walk_fn_ret_ty(&mut appears_in_output, &decl.output);

    debug!(?constrained_by_input.regions);

    // Walk the lifetimes that appear in where clauses.
    //
    // Subtle point: because we disallow nested bindings, we can just
    // ignore binders here and scrape up all names we see.
    let mut appears_in_where_clause = AllCollector::default();
    appears_in_where_clause.visit_generics(generics);

    for param in generics.params {
        if let hir::GenericParamKind::Lifetime { .. } = param.kind {
            if !param.bounds.is_empty() {
                // `'a: 'b` means both `'a` and `'b` are referenced
                appears_in_where_clause
                    .regions
                    .insert(hir::LifetimeName::Param(param.name.normalize_to_macros_2_0()));
            }
        }
    }

    debug!(?appears_in_where_clause.regions);

    // Late bound regions are those that:
    // - appear in the inputs
    // - do not appear in the where-clauses
    // - are not implicitly captured by `impl Trait`
    for param in generics.params {
        match param.kind {
            hir::GenericParamKind::Lifetime { .. } => { /* fall through */ }

            // Neither types nor consts are late-bound.
            hir::GenericParamKind::Type { .. } | hir::GenericParamKind::Const { .. } => continue,
        }

        let lt_name = hir::LifetimeName::Param(param.name.normalize_to_macros_2_0());
        // appears in the where clauses? early-bound.
        if appears_in_where_clause.regions.contains(&lt_name) {
            continue;
        }

        // does not appear in the inputs, but appears in the return type? early-bound.
        if !constrained_by_input.regions.contains(&lt_name)
            && appears_in_output.regions.contains(&lt_name)
        {
            continue;
        }

        debug!("lifetime {:?} with id {:?} is late-bound", param.name.ident(), param.hir_id);

        let inserted = map.late_bound.insert(param.hir_id);
        assert!(inserted, "visited lifetime {:?} twice", param.hir_id);
    }

    return;

    #[derive(Default)]
    struct ConstrainedCollector {
        regions: FxHashSet<hir::LifetimeName>,
    }

    impl<'v> Visitor<'v> for ConstrainedCollector {
        type Map = intravisit::ErasedMap<'v>;

        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
            NestedVisitorMap::None
        }

        fn visit_ty(&mut self, ty: &'v hir::Ty<'v>) {
            match ty.kind {
                hir::TyKind::Path(
                    hir::QPath::Resolved(Some(_), _) | hir::QPath::TypeRelative(..),
                ) => {
                    // ignore lifetimes appearing in associated type
                    // projections, as they are not *constrained*
                    // (defined above)
                }

                hir::TyKind::Path(hir::QPath::Resolved(None, ref path)) => {
                    // consider only the lifetimes on the final
                    // segment; I am not sure it's even currently
                    // valid to have them elsewhere, but even if it
                    // is, those would be potentially inputs to
                    // projections
                    if let Some(last_segment) = path.segments.last() {
                        self.visit_path_segment(path.span, last_segment);
                    }
                }

                _ => {
                    intravisit::walk_ty(self, ty);
                }
            }
        }

        fn visit_lifetime(&mut self, lifetime_ref: &'v hir::Lifetime) {
            self.regions.insert(lifetime_ref.name.normalize_to_macros_2_0());
        }
    }

    #[derive(Default)]
    struct AllCollector {
        regions: FxHashSet<hir::LifetimeName>,
    }

    impl<'v> Visitor<'v> for AllCollector {
        type Map = intravisit::ErasedMap<'v>;

        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
            NestedVisitorMap::None
        }

        fn visit_lifetime(&mut self, lifetime_ref: &'v hir::Lifetime) {
            self.regions.insert(lifetime_ref.name.normalize_to_macros_2_0());
        }
    }
}
