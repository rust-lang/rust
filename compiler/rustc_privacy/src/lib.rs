// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(associated_type_defaults)]
#![feature(rustdoc_internals)]
#![feature(try_blocks)]
// tidy-alphabetical-end

mod errors;

use std::fmt;
use std::marker::PhantomData;
use std::ops::ControlFlow;

use errors::{
    FieldIsPrivate, FieldIsPrivateLabel, FromPrivateDependencyInPublicInterface, InPublicInterface,
    ItemIsPrivate, PrivateInterfacesOrBoundsLint, ReportEffectiveVisibility, UnnameableTypesLint,
    UnnamedItemIsPrivate,
};
use rustc_ast::MacroDef;
use rustc_ast::visit::{VisitorResult, try_visit};
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::intern::Interned;
use rustc_errors::{MultiSpan, listify};
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{CRATE_DEF_ID, DefId, LocalDefId, LocalModDefId};
use rustc_hir::intravisit::{self, InferKind, Visitor};
use rustc_hir::{AmbigArg, AssocItemKind, ForeignItemKind, ItemId, ItemKind, PatKind};
use rustc_middle::middle::privacy::{EffectiveVisibilities, EffectiveVisibility, Level};
use rustc_middle::query::Providers;
use rustc_middle::ty::print::PrintTraitRefExt as _;
use rustc_middle::ty::{
    self, Const, GenericParamDefKind, TraitRef, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable,
    TypeVisitor,
};
use rustc_middle::{bug, span_bug};
use rustc_session::lint;
use rustc_span::hygiene::Transparency;
use rustc_span::{Ident, Span, Symbol, sym};
use tracing::debug;
use {rustc_attr_data_structures as attrs, rustc_hir as hir};

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

////////////////////////////////////////////////////////////////////////////////
/// Generic infrastructure used to implement specific visitors below.
////////////////////////////////////////////////////////////////////////////////

struct LazyDefPathStr<'tcx> {
    def_id: DefId,
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> fmt::Display for LazyDefPathStr<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tcx.def_path_str(self.def_id))
    }
}

/// Implemented to visit all `DefId`s in a type.
/// Visiting `DefId`s is useful because visibilities and reachabilities are attached to them.
/// The idea is to visit "all components of a type", as documented in
/// <https://github.com/rust-lang/rfcs/blob/master/text/2145-type-privacy.md#how-to-determine-visibility-of-a-type>.
/// The default type visitor (`TypeVisitor`) does most of the job, but it has some shortcomings.
/// First, it doesn't have overridable `fn visit_trait_ref`, so we have to catch trait `DefId`s
/// manually. Second, it doesn't visit some type components like signatures of fn types, or traits
/// in `impl Trait`, see individual comments in `DefIdVisitorSkeleton::visit_ty`.
pub trait DefIdVisitor<'tcx> {
    type Result: VisitorResult = ();
    const SHALLOW: bool = false;
    fn skip_assoc_tys(&self) -> bool {
        false
    }

    fn tcx(&self) -> TyCtxt<'tcx>;
    fn visit_def_id(&mut self, def_id: DefId, kind: &str, descr: &dyn fmt::Display)
    -> Self::Result;

    /// Not overridden, but used to actually visit types and traits.
    fn skeleton(&mut self) -> DefIdVisitorSkeleton<'_, 'tcx, Self> {
        DefIdVisitorSkeleton {
            def_id_visitor: self,
            visited_opaque_tys: Default::default(),
            dummy: Default::default(),
        }
    }
    fn visit(&mut self, ty_fragment: impl TypeVisitable<TyCtxt<'tcx>>) -> Self::Result {
        ty_fragment.visit_with(&mut self.skeleton())
    }
    fn visit_trait(&mut self, trait_ref: TraitRef<'tcx>) -> Self::Result {
        self.skeleton().visit_trait(trait_ref)
    }
    fn visit_predicates(&mut self, predicates: ty::GenericPredicates<'tcx>) -> Self::Result {
        self.skeleton().visit_clauses(predicates.predicates)
    }
    fn visit_clauses(&mut self, clauses: &[(ty::Clause<'tcx>, Span)]) -> Self::Result {
        self.skeleton().visit_clauses(clauses)
    }
}

pub struct DefIdVisitorSkeleton<'v, 'tcx, V: ?Sized> {
    def_id_visitor: &'v mut V,
    visited_opaque_tys: FxHashSet<DefId>,
    dummy: PhantomData<TyCtxt<'tcx>>,
}

impl<'tcx, V> DefIdVisitorSkeleton<'_, 'tcx, V>
where
    V: DefIdVisitor<'tcx> + ?Sized,
{
    fn visit_trait(&mut self, trait_ref: TraitRef<'tcx>) -> V::Result {
        let TraitRef { def_id, args, .. } = trait_ref;
        try_visit!(self.def_id_visitor.visit_def_id(
            def_id,
            "trait",
            &trait_ref.print_only_trait_path()
        ));
        if V::SHALLOW { V::Result::output() } else { args.visit_with(self) }
    }

    fn visit_projection_term(&mut self, projection: ty::AliasTerm<'tcx>) -> V::Result {
        let tcx = self.def_id_visitor.tcx();
        let (trait_ref, assoc_args) = projection.trait_ref_and_own_args(tcx);
        try_visit!(self.visit_trait(trait_ref));
        if V::SHALLOW {
            V::Result::output()
        } else {
            V::Result::from_branch(
                assoc_args.iter().try_for_each(|arg| arg.visit_with(self).branch()),
            )
        }
    }

    fn visit_clause(&mut self, clause: ty::Clause<'tcx>) -> V::Result {
        match clause.kind().skip_binder() {
            ty::ClauseKind::Trait(ty::TraitPredicate { trait_ref, polarity: _ }) => {
                self.visit_trait(trait_ref)
            }
            ty::ClauseKind::HostEffect(pred) => {
                try_visit!(self.visit_trait(pred.trait_ref));
                pred.constness.visit_with(self)
            }
            ty::ClauseKind::Projection(ty::ProjectionPredicate {
                projection_term: projection_ty,
                term,
            }) => {
                try_visit!(term.visit_with(self));
                self.visit_projection_term(projection_ty)
            }
            ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(ty, _region)) => ty.visit_with(self),
            ty::ClauseKind::RegionOutlives(..) => V::Result::output(),
            ty::ClauseKind::ConstArgHasType(ct, ty) => {
                try_visit!(ct.visit_with(self));
                ty.visit_with(self)
            }
            ty::ClauseKind::ConstEvaluatable(ct) => ct.visit_with(self),
            ty::ClauseKind::WellFormed(term) => term.visit_with(self),
        }
    }

    fn visit_clauses(&mut self, clauses: &[(ty::Clause<'tcx>, Span)]) -> V::Result {
        for &(clause, _) in clauses {
            try_visit!(self.visit_clause(clause));
        }
        V::Result::output()
    }
}

impl<'tcx, V> TypeVisitor<TyCtxt<'tcx>> for DefIdVisitorSkeleton<'_, 'tcx, V>
where
    V: DefIdVisitor<'tcx> + ?Sized,
{
    type Result = V::Result;

    fn visit_predicate(&mut self, p: ty::Predicate<'tcx>) -> Self::Result {
        self.visit_clause(p.as_clause().unwrap())
    }

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
        let tcx = self.def_id_visitor.tcx();
        // GenericArgs are not visited here because they are visited below
        // in `super_visit_with`.
        match *ty.kind() {
            ty::Adt(ty::AdtDef(Interned(&ty::AdtDefData { did: def_id, .. }, _)), ..)
            | ty::Foreign(def_id)
            | ty::FnDef(def_id, ..)
            | ty::Closure(def_id, ..)
            | ty::CoroutineClosure(def_id, ..)
            | ty::Coroutine(def_id, ..) => {
                try_visit!(self.def_id_visitor.visit_def_id(def_id, "type", &ty));
                if V::SHALLOW {
                    return V::Result::output();
                }
                // Default type visitor doesn't visit signatures of fn types.
                // Something like `fn() -> Priv {my_func}` is considered a private type even if
                // `my_func` is public, so we need to visit signatures.
                if let ty::FnDef(..) = ty.kind() {
                    // FIXME: this should probably use `args` from `FnDef`
                    try_visit!(tcx.fn_sig(def_id).instantiate_identity().visit_with(self));
                }
                // Inherent static methods don't have self type in args.
                // Something like `fn() {my_method}` type of the method
                // `impl Pub<Priv> { pub fn my_method() {} }` is considered a private type,
                // so we need to visit the self type additionally.
                if let Some(assoc_item) = tcx.opt_associated_item(def_id) {
                    if let Some(impl_def_id) = assoc_item.impl_container(tcx) {
                        try_visit!(
                            tcx.type_of(impl_def_id).instantiate_identity().visit_with(self)
                        );
                    }
                }
            }
            ty::Alias(kind @ (ty::Inherent | ty::Free | ty::Projection), data) => {
                if self.def_id_visitor.skip_assoc_tys() {
                    // Visitors searching for minimal visibility/reachability want to
                    // conservatively approximate associated types like `Type::Alias`
                    // as visible/reachable even if `Type` is private.
                    // Ideally, associated types should be instantiated in the same way as
                    // free type aliases, but this isn't done yet.
                    return V::Result::output();
                }

                try_visit!(self.def_id_visitor.visit_def_id(
                    data.def_id,
                    match kind {
                        ty::Inherent | ty::Projection => "associated type",
                        ty::Free => "type alias",
                        ty::Opaque => unreachable!(),
                    },
                    &LazyDefPathStr { def_id: data.def_id, tcx },
                ));

                // This will also visit args if necessary, so we don't need to recurse.
                return if V::SHALLOW {
                    V::Result::output()
                } else if kind == ty::Projection {
                    self.visit_projection_term(data.into())
                } else {
                    V::Result::from_branch(
                        data.args.iter().try_for_each(|arg| arg.visit_with(self).branch()),
                    )
                };
            }
            ty::Dynamic(predicates, ..) => {
                // All traits in the list are considered the "primary" part of the type
                // and are visited by shallow visitors.
                for predicate in predicates {
                    let trait_ref = match predicate.skip_binder() {
                        ty::ExistentialPredicate::Trait(trait_ref) => trait_ref,
                        ty::ExistentialPredicate::Projection(proj) => proj.trait_ref(tcx),
                        ty::ExistentialPredicate::AutoTrait(def_id) => {
                            ty::ExistentialTraitRef::new(tcx, def_id, ty::GenericArgs::empty())
                        }
                    };
                    let ty::ExistentialTraitRef { def_id, .. } = trait_ref;
                    try_visit!(self.def_id_visitor.visit_def_id(def_id, "trait", &trait_ref));
                }
            }
            ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }) => {
                // Skip repeated `Opaque`s to avoid infinite recursion.
                if self.visited_opaque_tys.insert(def_id) {
                    // The intent is to treat `impl Trait1 + Trait2` identically to
                    // `dyn Trait1 + Trait2`. Therefore we ignore def-id of the opaque type itself
                    // (it either has no visibility, or its visibility is insignificant, like
                    // visibilities of type aliases) and recurse into bounds instead to go
                    // through the trait list (default type visitor doesn't visit those traits).
                    // All traits in the list are considered the "primary" part of the type
                    // and are visited by shallow visitors.
                    try_visit!(self.visit_clauses(tcx.explicit_item_bounds(def_id).skip_binder()));
                }
            }
            // These types don't have their own def-ids (but may have subcomponents
            // with def-ids that should be visited recursively).
            ty::Bool
            | ty::Char
            | ty::Int(..)
            | ty::Uint(..)
            | ty::Float(..)
            | ty::Str
            | ty::Never
            | ty::Array(..)
            | ty::Slice(..)
            | ty::Tuple(..)
            | ty::RawPtr(..)
            | ty::Ref(..)
            | ty::Pat(..)
            | ty::FnPtr(..)
            | ty::UnsafeBinder(_)
            | ty::Param(..)
            | ty::Bound(..)
            | ty::Error(_)
            | ty::CoroutineWitness(..) => {}
            ty::Placeholder(..) | ty::Infer(..) => {
                bug!("unexpected type: {:?}", ty)
            }
        }

        if V::SHALLOW { V::Result::output() } else { ty.super_visit_with(self) }
    }

    fn visit_const(&mut self, c: Const<'tcx>) -> Self::Result {
        let tcx = self.def_id_visitor.tcx();
        tcx.expand_abstract_consts(c).super_visit_with(self)
    }
}

fn min(vis1: ty::Visibility, vis2: ty::Visibility, tcx: TyCtxt<'_>) -> ty::Visibility {
    if vis1.is_at_least(vis2, tcx) { vis2 } else { vis1 }
}

////////////////////////////////////////////////////////////////////////////////
/// Visitor used to determine impl visibility and reachability.
////////////////////////////////////////////////////////////////////////////////

struct FindMin<'a, 'tcx, VL: VisibilityLike, const SHALLOW: bool> {
    tcx: TyCtxt<'tcx>,
    effective_visibilities: &'a EffectiveVisibilities,
    min: VL,
}

impl<'a, 'tcx, VL: VisibilityLike, const SHALLOW: bool> DefIdVisitor<'tcx>
    for FindMin<'a, 'tcx, VL, SHALLOW>
{
    const SHALLOW: bool = SHALLOW;
    fn skip_assoc_tys(&self) -> bool {
        true
    }
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
    fn visit_def_id(&mut self, def_id: DefId, _kind: &str, _descr: &dyn fmt::Display) {
        if let Some(def_id) = def_id.as_local() {
            self.min = VL::new_min(self, def_id);
        }
    }
}

trait VisibilityLike: Sized {
    const MAX: Self;
    fn new_min<const SHALLOW: bool>(
        find: &FindMin<'_, '_, Self, SHALLOW>,
        def_id: LocalDefId,
    ) -> Self;

    // Returns an over-approximation (`skip_assoc_tys()` = true) of visibility due to
    // associated types for which we can't determine visibility precisely.
    fn of_impl<const SHALLOW: bool>(
        def_id: LocalDefId,
        tcx: TyCtxt<'_>,
        effective_visibilities: &EffectiveVisibilities,
    ) -> Self {
        let mut find = FindMin::<_, SHALLOW> { tcx, effective_visibilities, min: Self::MAX };
        find.visit(tcx.type_of(def_id).instantiate_identity());
        if let Some(trait_ref) = tcx.impl_trait_ref(def_id) {
            find.visit_trait(trait_ref.instantiate_identity());
        }
        find.min
    }
}

impl VisibilityLike for ty::Visibility {
    const MAX: Self = ty::Visibility::Public;
    fn new_min<const SHALLOW: bool>(
        find: &FindMin<'_, '_, Self, SHALLOW>,
        def_id: LocalDefId,
    ) -> Self {
        min(find.tcx.local_visibility(def_id), find.min, find.tcx)
    }
}

impl VisibilityLike for EffectiveVisibility {
    const MAX: Self = EffectiveVisibility::from_vis(ty::Visibility::Public);
    fn new_min<const SHALLOW: bool>(
        find: &FindMin<'_, '_, Self, SHALLOW>,
        def_id: LocalDefId,
    ) -> Self {
        let effective_vis =
            find.effective_visibilities.effective_vis(def_id).copied().unwrap_or_else(|| {
                let private_vis = ty::Visibility::Restricted(
                    find.tcx.parent_module_from_def_id(def_id).to_local_def_id(),
                );
                EffectiveVisibility::from_vis(private_vis)
            });

        effective_vis.min(find.min, find.tcx)
    }
}

////////////////////////////////////////////////////////////////////////////////
/// The embargo visitor, used to determine the exports of the AST.
////////////////////////////////////////////////////////////////////////////////

struct EmbargoVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,

    /// Effective visibilities for reachable nodes.
    effective_visibilities: EffectiveVisibilities,
    /// A set of pairs corresponding to modules, where the first module is
    /// reachable via a macro that's defined in the second module. This cannot
    /// be represented as reachable because it can't handle the following case:
    ///
    /// pub mod n {                         // Should be `Public`
    ///     pub(crate) mod p {              // Should *not* be accessible
    ///         pub fn f() -> i32 { 12 }    // Must be `Reachable`
    ///     }
    /// }
    /// pub macro m() {
    ///     n::p::f()
    /// }
    macro_reachable: FxHashSet<(LocalModDefId, LocalModDefId)>,
    /// Has something changed in the level map?
    changed: bool,
}

struct ReachEverythingInTheInterfaceVisitor<'a, 'tcx> {
    effective_vis: EffectiveVisibility,
    item_def_id: LocalDefId,
    ev: &'a mut EmbargoVisitor<'tcx>,
    level: Level,
}

impl<'tcx> EmbargoVisitor<'tcx> {
    fn get(&self, def_id: LocalDefId) -> Option<EffectiveVisibility> {
        self.effective_visibilities.effective_vis(def_id).copied()
    }

    // Updates node effective visibility.
    fn update(
        &mut self,
        def_id: LocalDefId,
        inherited_effective_vis: EffectiveVisibility,
        level: Level,
    ) {
        let nominal_vis = self.tcx.local_visibility(def_id);
        self.update_eff_vis(def_id, inherited_effective_vis, Some(nominal_vis), level);
    }

    fn update_eff_vis(
        &mut self,
        def_id: LocalDefId,
        inherited_effective_vis: EffectiveVisibility,
        max_vis: Option<ty::Visibility>,
        level: Level,
    ) {
        // FIXME(typed_def_id): Make `Visibility::Restricted` use a `LocalModDefId` by default.
        let private_vis =
            ty::Visibility::Restricted(self.tcx.parent_module_from_def_id(def_id).into());
        if max_vis != Some(private_vis) {
            self.changed |= self.effective_visibilities.update(
                def_id,
                max_vis,
                || private_vis,
                inherited_effective_vis,
                level,
                self.tcx,
            );
        }
    }

    fn reach(
        &mut self,
        def_id: LocalDefId,
        effective_vis: EffectiveVisibility,
    ) -> ReachEverythingInTheInterfaceVisitor<'_, 'tcx> {
        ReachEverythingInTheInterfaceVisitor {
            effective_vis,
            item_def_id: def_id,
            ev: self,
            level: Level::Reachable,
        }
    }

    fn reach_through_impl_trait(
        &mut self,
        def_id: LocalDefId,
        effective_vis: EffectiveVisibility,
    ) -> ReachEverythingInTheInterfaceVisitor<'_, 'tcx> {
        ReachEverythingInTheInterfaceVisitor {
            effective_vis,
            item_def_id: def_id,
            ev: self,
            level: Level::ReachableThroughImplTrait,
        }
    }

    // We have to make sure that the items that macros might reference
    // are reachable, since they might be exported transitively.
    fn update_reachability_from_macro(
        &mut self,
        local_def_id: LocalDefId,
        md: &MacroDef,
        macro_ev: EffectiveVisibility,
    ) {
        // Non-opaque macros cannot make other items more accessible than they already are.
        let hir_id = self.tcx.local_def_id_to_hir_id(local_def_id);
        let attrs = self.tcx.hir_attrs(hir_id);

        if attrs::find_attr!(attrs, attrs::AttributeKind::MacroTransparency(x) => *x)
            .unwrap_or(Transparency::fallback(md.macro_rules))
            != Transparency::Opaque
        {
            return;
        }

        let macro_module_def_id = self.tcx.local_parent(local_def_id);
        if self.tcx.def_kind(macro_module_def_id) != DefKind::Mod {
            // The macro's parent doesn't correspond to a `mod`, return early (#63164, #65252).
            return;
        }
        // FIXME(typed_def_id): Introduce checked constructors that check def_kind.
        let macro_module_def_id = LocalModDefId::new_unchecked(macro_module_def_id);

        if self.effective_visibilities.public_at_level(local_def_id).is_none() {
            return;
        }

        // Since we are starting from an externally visible module,
        // all the parents in the loop below are also guaranteed to be modules.
        let mut module_def_id = macro_module_def_id;
        loop {
            let changed_reachability =
                self.update_macro_reachable(module_def_id, macro_module_def_id, macro_ev);
            if changed_reachability || module_def_id == LocalModDefId::CRATE_DEF_ID {
                break;
            }
            module_def_id = LocalModDefId::new_unchecked(self.tcx.local_parent(module_def_id));
        }
    }

    /// Updates the item as being reachable through a macro defined in the given
    /// module. Returns `true` if the level has changed.
    fn update_macro_reachable(
        &mut self,
        module_def_id: LocalModDefId,
        defining_mod: LocalModDefId,
        macro_ev: EffectiveVisibility,
    ) -> bool {
        if self.macro_reachable.insert((module_def_id, defining_mod)) {
            for child in self.tcx.module_children_local(module_def_id.to_local_def_id()) {
                if let Res::Def(def_kind, def_id) = child.res
                    && let Some(def_id) = def_id.as_local()
                    && child.vis.is_accessible_from(defining_mod, self.tcx)
                {
                    let vis = self.tcx.local_visibility(def_id);
                    self.update_macro_reachable_def(def_id, def_kind, vis, defining_mod, macro_ev);
                }
            }
            true
        } else {
            false
        }
    }

    fn update_macro_reachable_def(
        &mut self,
        def_id: LocalDefId,
        def_kind: DefKind,
        vis: ty::Visibility,
        module: LocalModDefId,
        macro_ev: EffectiveVisibility,
    ) {
        self.update(def_id, macro_ev, Level::Reachable);
        match def_kind {
            // No type privacy, so can be directly marked as reachable.
            DefKind::Const | DefKind::Static { .. } | DefKind::TraitAlias | DefKind::TyAlias => {
                if vis.is_accessible_from(module, self.tcx) {
                    self.update(def_id, macro_ev, Level::Reachable);
                }
            }

            // Hygiene isn't really implemented for `macro_rules!` macros at the
            // moment. Accordingly, marking them as reachable is unwise. `macro` macros
            // have normal hygiene, so we can treat them like other items without type
            // privacy and mark them reachable.
            DefKind::Macro(_) => {
                let item = self.tcx.hir_expect_item(def_id);
                if let hir::ItemKind::Macro(_, MacroDef { macro_rules: false, .. }, _) = item.kind {
                    if vis.is_accessible_from(module, self.tcx) {
                        self.update(def_id, macro_ev, Level::Reachable);
                    }
                }
            }

            // We can't use a module name as the final segment of a path, except
            // in use statements. Since re-export checking doesn't consider
            // hygiene these don't need to be marked reachable. The contents of
            // the module, however may be reachable.
            DefKind::Mod => {
                if vis.is_accessible_from(module, self.tcx) {
                    self.update_macro_reachable(
                        LocalModDefId::new_unchecked(def_id),
                        module,
                        macro_ev,
                    );
                }
            }

            DefKind::Struct | DefKind::Union => {
                // While structs and unions have type privacy, their fields do not.
                let item = self.tcx.hir_expect_item(def_id);
                if let hir::ItemKind::Struct(_, ref struct_def, _)
                | hir::ItemKind::Union(_, ref struct_def, _) = item.kind
                {
                    for field in struct_def.fields() {
                        let field_vis = self.tcx.local_visibility(field.def_id);
                        if field_vis.is_accessible_from(module, self.tcx) {
                            self.reach(field.def_id, macro_ev).ty();
                        }
                    }
                } else {
                    bug!("item {:?} with DefKind {:?}", item, def_kind);
                }
            }

            // These have type privacy, so are not reachable unless they're
            // public, or are not namespaced at all.
            DefKind::AssocConst
            | DefKind::AssocTy
            | DefKind::ConstParam
            | DefKind::Ctor(_, _)
            | DefKind::Enum
            | DefKind::ForeignTy
            | DefKind::Fn
            | DefKind::OpaqueTy
            | DefKind::AssocFn
            | DefKind::Trait
            | DefKind::TyParam
            | DefKind::Variant
            | DefKind::LifetimeParam
            | DefKind::ExternCrate
            | DefKind::Use
            | DefKind::ForeignMod
            | DefKind::AnonConst
            | DefKind::InlineConst
            | DefKind::Field
            | DefKind::GlobalAsm
            | DefKind::Impl { .. }
            | DefKind::Closure
            | DefKind::SyntheticCoroutineBody => (),
        }
    }
}

impl<'tcx> Visitor<'tcx> for EmbargoVisitor<'tcx> {
    fn visit_item(&mut self, item: &'tcx hir::Item<'tcx>) {
        // Update levels of nested things and mark all items
        // in interfaces of reachable items as reachable.
        let item_ev = self.get(item.owner_id.def_id);
        match item.kind {
            // The interface is empty, and no nested items.
            hir::ItemKind::Use(..)
            | hir::ItemKind::ExternCrate(..)
            | hir::ItemKind::GlobalAsm { .. } => {}
            // The interface is empty, and all nested items are processed by `visit_item`.
            hir::ItemKind::Mod(..) => {}
            hir::ItemKind::Macro(_, macro_def, _) => {
                if let Some(item_ev) = item_ev {
                    self.update_reachability_from_macro(item.owner_id.def_id, macro_def, item_ev);
                }
            }
            hir::ItemKind::Const(..)
            | hir::ItemKind::Static(..)
            | hir::ItemKind::Fn { .. }
            | hir::ItemKind::TyAlias(..) => {
                if let Some(item_ev) = item_ev {
                    self.reach(item.owner_id.def_id, item_ev).generics().predicates().ty();
                }
            }
            hir::ItemKind::Trait(.., trait_item_refs) => {
                if let Some(item_ev) = item_ev {
                    self.reach(item.owner_id.def_id, item_ev).generics().predicates();

                    for trait_item_ref in trait_item_refs {
                        self.update(trait_item_ref.id.owner_id.def_id, item_ev, Level::Reachable);

                        let tcx = self.tcx;
                        let mut reach = self.reach(trait_item_ref.id.owner_id.def_id, item_ev);
                        reach.generics().predicates();

                        if trait_item_ref.kind == AssocItemKind::Type
                            && !tcx.defaultness(trait_item_ref.id.owner_id).has_value()
                        {
                            // No type to visit.
                        } else {
                            reach.ty();
                        }
                    }
                }
            }
            hir::ItemKind::TraitAlias(..) => {
                if let Some(item_ev) = item_ev {
                    self.reach(item.owner_id.def_id, item_ev).generics().predicates();
                }
            }
            hir::ItemKind::Impl(impl_) => {
                // Type inference is very smart sometimes. It can make an impl reachable even some
                // components of its type or trait are unreachable. E.g. methods of
                // `impl ReachableTrait<UnreachableTy> for ReachableTy<UnreachableTy> { ... }`
                // can be usable from other crates (#57264). So we skip args when calculating
                // reachability and consider an impl reachable if its "shallow" type and trait are
                // reachable.
                //
                // The assumption we make here is that type-inference won't let you use an impl
                // without knowing both "shallow" version of its self type and "shallow" version of
                // its trait if it exists (which require reaching the `DefId`s in them).
                let item_ev = EffectiveVisibility::of_impl::<true>(
                    item.owner_id.def_id,
                    self.tcx,
                    &self.effective_visibilities,
                );

                self.update_eff_vis(item.owner_id.def_id, item_ev, None, Level::Direct);

                self.reach(item.owner_id.def_id, item_ev).generics().predicates().ty().trait_ref();

                for impl_item_ref in impl_.items {
                    let def_id = impl_item_ref.id.owner_id.def_id;
                    let max_vis =
                        impl_.of_trait.is_none().then(|| self.tcx.local_visibility(def_id));
                    self.update_eff_vis(def_id, item_ev, max_vis, Level::Direct);

                    if let Some(impl_item_ev) = self.get(def_id) {
                        self.reach(def_id, impl_item_ev).generics().predicates().ty();
                    }
                }
            }
            hir::ItemKind::Enum(_, ref def, _) => {
                if let Some(item_ev) = item_ev {
                    self.reach(item.owner_id.def_id, item_ev).generics().predicates();
                }
                for variant in def.variants {
                    if let Some(item_ev) = item_ev {
                        self.update(variant.def_id, item_ev, Level::Reachable);
                    }

                    if let Some(variant_ev) = self.get(variant.def_id) {
                        if let Some(ctor_def_id) = variant.data.ctor_def_id() {
                            self.update(ctor_def_id, variant_ev, Level::Reachable);
                        }
                        for field in variant.data.fields() {
                            self.update(field.def_id, variant_ev, Level::Reachable);
                            self.reach(field.def_id, variant_ev).ty();
                        }
                        // Corner case: if the variant is reachable, but its
                        // enum is not, make the enum reachable as well.
                        self.reach(item.owner_id.def_id, variant_ev).ty();
                    }
                    if let Some(ctor_def_id) = variant.data.ctor_def_id() {
                        if let Some(ctor_ev) = self.get(ctor_def_id) {
                            self.reach(item.owner_id.def_id, ctor_ev).ty();
                        }
                    }
                }
            }
            hir::ItemKind::ForeignMod { items, .. } => {
                for foreign_item in items {
                    if let Some(foreign_item_ev) = self.get(foreign_item.id.owner_id.def_id) {
                        self.reach(foreign_item.id.owner_id.def_id, foreign_item_ev)
                            .generics()
                            .predicates()
                            .ty();
                    }
                }
            }
            hir::ItemKind::Struct(_, ref struct_def, _)
            | hir::ItemKind::Union(_, ref struct_def, _) => {
                if let Some(item_ev) = item_ev {
                    self.reach(item.owner_id.def_id, item_ev).generics().predicates();
                    for field in struct_def.fields() {
                        self.update(field.def_id, item_ev, Level::Reachable);
                        if let Some(field_ev) = self.get(field.def_id) {
                            self.reach(field.def_id, field_ev).ty();
                        }
                    }
                }
                if let Some(ctor_def_id) = struct_def.ctor_def_id() {
                    if let Some(item_ev) = item_ev {
                        self.update(ctor_def_id, item_ev, Level::Reachable);
                    }
                    if let Some(ctor_ev) = self.get(ctor_def_id) {
                        self.reach(item.owner_id.def_id, ctor_ev).ty();
                    }
                }
            }
        }
    }
}

impl ReachEverythingInTheInterfaceVisitor<'_, '_> {
    fn generics(&mut self) -> &mut Self {
        for param in &self.ev.tcx.generics_of(self.item_def_id).own_params {
            if let GenericParamDefKind::Const { .. } = param.kind {
                self.visit(self.ev.tcx.type_of(param.def_id).instantiate_identity());
            }
            if let Some(default) = param.default_value(self.ev.tcx) {
                self.visit(default.instantiate_identity());
            }
        }
        self
    }

    fn predicates(&mut self) -> &mut Self {
        self.visit_predicates(self.ev.tcx.predicates_of(self.item_def_id));
        self
    }

    fn ty(&mut self) -> &mut Self {
        self.visit(self.ev.tcx.type_of(self.item_def_id).instantiate_identity());
        self
    }

    fn trait_ref(&mut self) -> &mut Self {
        if let Some(trait_ref) = self.ev.tcx.impl_trait_ref(self.item_def_id) {
            self.visit_trait(trait_ref.instantiate_identity());
        }
        self
    }
}

impl<'tcx> DefIdVisitor<'tcx> for ReachEverythingInTheInterfaceVisitor<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.ev.tcx
    }
    fn visit_def_id(&mut self, def_id: DefId, _kind: &str, _descr: &dyn fmt::Display) {
        if let Some(def_id) = def_id.as_local() {
            // All effective visibilities except `reachable_through_impl_trait` are limited to
            // nominal visibility. If any type or trait is leaked farther than that, it will
            // produce type privacy errors on any use, so we don't consider it leaked.
            let max_vis = (self.level != Level::ReachableThroughImplTrait)
                .then(|| self.ev.tcx.local_visibility(def_id));
            self.ev.update_eff_vis(def_id, self.effective_vis, max_vis, self.level);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Visitor, used for EffectiveVisibilities table checking
////////////////////////////////////////////////////////////////////////////////
pub struct TestReachabilityVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    effective_visibilities: &'a EffectiveVisibilities,
}

impl<'a, 'tcx> TestReachabilityVisitor<'a, 'tcx> {
    fn effective_visibility_diagnostic(&mut self, def_id: LocalDefId) {
        if self.tcx.has_attr(def_id, sym::rustc_effective_visibility) {
            let mut error_msg = String::new();
            let span = self.tcx.def_span(def_id.to_def_id());
            if let Some(effective_vis) = self.effective_visibilities.effective_vis(def_id) {
                for level in Level::all_levels() {
                    let vis_str = effective_vis.at_level(level).to_string(def_id, self.tcx);
                    if level != Level::Direct {
                        error_msg.push_str(", ");
                    }
                    error_msg.push_str(&format!("{level:?}: {vis_str}"));
                }
            } else {
                error_msg.push_str("not in the table");
            }
            self.tcx.dcx().emit_err(ReportEffectiveVisibility { span, descr: error_msg });
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for TestReachabilityVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &'tcx hir::Item<'tcx>) {
        self.effective_visibility_diagnostic(item.owner_id.def_id);

        match item.kind {
            hir::ItemKind::Enum(_, ref def, _) => {
                for variant in def.variants.iter() {
                    self.effective_visibility_diagnostic(variant.def_id);
                    if let Some(ctor_def_id) = variant.data.ctor_def_id() {
                        self.effective_visibility_diagnostic(ctor_def_id);
                    }
                    for field in variant.data.fields() {
                        self.effective_visibility_diagnostic(field.def_id);
                    }
                }
            }
            hir::ItemKind::Struct(_, ref def, _) | hir::ItemKind::Union(_, ref def, _) => {
                if let Some(ctor_def_id) = def.ctor_def_id() {
                    self.effective_visibility_diagnostic(ctor_def_id);
                }
                for field in def.fields() {
                    self.effective_visibility_diagnostic(field.def_id);
                }
            }
            _ => {}
        }
    }

    fn visit_trait_item(&mut self, item: &'tcx hir::TraitItem<'tcx>) {
        self.effective_visibility_diagnostic(item.owner_id.def_id);
    }
    fn visit_impl_item(&mut self, item: &'tcx hir::ImplItem<'tcx>) {
        self.effective_visibility_diagnostic(item.owner_id.def_id);
    }
    fn visit_foreign_item(&mut self, item: &'tcx hir::ForeignItem<'tcx>) {
        self.effective_visibility_diagnostic(item.owner_id.def_id);
    }
}

//////////////////////////////////////////////////////////////////////////////////////
/// Name privacy visitor, checks privacy and reports violations.
/// Most of name privacy checks are performed during the main resolution phase,
/// or later in type checking when field accesses and associated items are resolved.
/// This pass performs remaining checks for fields in struct expressions and patterns.
//////////////////////////////////////////////////////////////////////////////////////

struct NamePrivacyVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    maybe_typeck_results: Option<&'tcx ty::TypeckResults<'tcx>>,
}

impl<'tcx> NamePrivacyVisitor<'tcx> {
    /// Gets the type-checking results for the current body.
    /// As this will ICE if called outside bodies, only call when working with
    /// `Expr` or `Pat` nodes (they are guaranteed to be found only in bodies).
    #[track_caller]
    fn typeck_results(&self) -> &'tcx ty::TypeckResults<'tcx> {
        self.maybe_typeck_results
            .expect("`NamePrivacyVisitor::typeck_results` called outside of body")
    }

    // Checks that a field in a struct constructor (expression or pattern) is accessible.
    fn check_field(
        &mut self,
        hir_id: hir::HirId,    // ID of the field use
        use_ctxt: Span,        // syntax context of the field name at the use site
        def: ty::AdtDef<'tcx>, // definition of the struct or enum
        field: &'tcx ty::FieldDef,
    ) -> bool {
        if def.is_enum() {
            return true;
        }

        // definition of the field
        let ident = Ident::new(sym::dummy, use_ctxt);
        let (_, def_id) = self.tcx.adjust_ident_and_get_scope(ident, def.did(), hir_id);
        !field.vis.is_accessible_from(def_id, self.tcx)
    }

    // Checks that a field in a struct constructor (expression or pattern) is accessible.
    fn emit_unreachable_field_error(
        &mut self,
        fields: Vec<(Symbol, Span, bool /* field is present */)>,
        def: ty::AdtDef<'tcx>, // definition of the struct or enum
        update_syntax: Option<Span>,
        struct_span: Span,
    ) {
        if def.is_enum() || fields.is_empty() {
            return;
        }

        //   error[E0451]: fields `beta` and `gamma` of struct `Alpha` are private
        //   --> $DIR/visibility.rs:18:13
        //    |
        // LL |     let _x = Alpha {
        //    |              ----- in this type      # from `def`
        // LL |         beta: 0,
        //    |         ^^^^^^^ private field        # `fields.2` is `true`
        // LL |         ..
        //    |         ^^ field `gamma` is private  # `fields.2` is `false`

        // Get the list of all private fields for the main message.
        let Some(field_names) = listify(&fields[..], |(n, _, _)| format!("`{n}`")) else { return };
        let span: MultiSpan = fields.iter().map(|(_, span, _)| *span).collect::<Vec<Span>>().into();

        // Get the list of all private fields when pointing at the `..rest`.
        let rest_field_names: Vec<_> =
            fields.iter().filter(|(_, _, is_present)| !is_present).map(|(n, _, _)| n).collect();
        let rest_len = rest_field_names.len();
        let rest_field_names =
            listify(&rest_field_names[..], |n| format!("`{n}`")).unwrap_or_default();
        // Get all the labels for each field or `..rest` in the primary MultiSpan.
        let labels = fields
            .iter()
            .filter(|(_, _, is_present)| *is_present)
            .map(|(_, span, _)| FieldIsPrivateLabel::Other { span: *span })
            .chain(update_syntax.iter().map(|span| FieldIsPrivateLabel::IsUpdateSyntax {
                span: *span,
                rest_field_names: rest_field_names.clone(),
                rest_len,
            }))
            .collect();

        self.tcx.dcx().emit_err(FieldIsPrivate {
            span,
            struct_span: if self
                .tcx
                .sess
                .source_map()
                .is_multiline(fields[0].1.between(struct_span))
            {
                Some(struct_span)
            } else {
                None
            },
            field_names,
            variant_descr: def.variant_descr(),
            def_path_str: self.tcx.def_path_str(def.did()),
            labels,
            len: fields.len(),
        });
    }

    fn check_expanded_fields(
        &mut self,
        adt: ty::AdtDef<'tcx>,
        variant: &'tcx ty::VariantDef,
        fields: &[hir::ExprField<'tcx>],
        hir_id: hir::HirId,
        span: Span,
        struct_span: Span,
    ) {
        let mut failed_fields = vec![];
        for (vf_index, variant_field) in variant.fields.iter_enumerated() {
            let field =
                fields.iter().find(|f| self.typeck_results().field_index(f.hir_id) == vf_index);
            let (hir_id, use_ctxt, span) = match field {
                Some(field) => (field.hir_id, field.ident.span, field.span),
                None => (hir_id, span, span),
            };
            if self.check_field(hir_id, use_ctxt, adt, variant_field) {
                let name = match field {
                    Some(field) => field.ident.name,
                    None => variant_field.name,
                };
                failed_fields.push((name, span, field.is_some()));
            }
        }
        self.emit_unreachable_field_error(failed_fields, adt, Some(span), struct_span);
    }
}

impl<'tcx> Visitor<'tcx> for NamePrivacyVisitor<'tcx> {
    fn visit_nested_body(&mut self, body_id: hir::BodyId) {
        let new_typeck_results = self.tcx.typeck_body(body_id);
        // Do not try reporting privacy violations if we failed to infer types.
        if new_typeck_results.tainted_by_errors.is_some() {
            return;
        }
        let old_maybe_typeck_results = self.maybe_typeck_results.replace(new_typeck_results);
        self.visit_body(self.tcx.hir_body(body_id));
        self.maybe_typeck_results = old_maybe_typeck_results;
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Struct(qpath, fields, ref base) = expr.kind {
            let res = self.typeck_results().qpath_res(qpath, expr.hir_id);
            let adt = self.typeck_results().expr_ty(expr).ty_adt_def().unwrap();
            let variant = adt.variant_of_res(res);
            match *base {
                hir::StructTailExpr::Base(base) => {
                    // If the expression uses FRU we need to make sure all the unmentioned fields
                    // are checked for privacy (RFC 736). Rather than computing the set of
                    // unmentioned fields, just check them all.
                    self.check_expanded_fields(
                        adt,
                        variant,
                        fields,
                        base.hir_id,
                        base.span,
                        qpath.span(),
                    );
                }
                hir::StructTailExpr::DefaultFields(span) => {
                    self.check_expanded_fields(
                        adt,
                        variant,
                        fields,
                        expr.hir_id,
                        span,
                        qpath.span(),
                    );
                }
                hir::StructTailExpr::None => {
                    let mut failed_fields = vec![];
                    for field in fields {
                        let (hir_id, use_ctxt) = (field.hir_id, field.ident.span);
                        let index = self.typeck_results().field_index(field.hir_id);
                        if self.check_field(hir_id, use_ctxt, adt, &variant.fields[index]) {
                            failed_fields.push((field.ident.name, field.ident.span, true));
                        }
                    }
                    self.emit_unreachable_field_error(failed_fields, adt, None, qpath.span());
                }
            }
        }

        intravisit::walk_expr(self, expr);
    }

    fn visit_pat(&mut self, pat: &'tcx hir::Pat<'tcx>) {
        if let PatKind::Struct(ref qpath, fields, _) = pat.kind {
            let res = self.typeck_results().qpath_res(qpath, pat.hir_id);
            let adt = self.typeck_results().pat_ty(pat).ty_adt_def().unwrap();
            let variant = adt.variant_of_res(res);
            let mut failed_fields = vec![];
            for field in fields {
                let (hir_id, use_ctxt) = (field.hir_id, field.ident.span);
                let index = self.typeck_results().field_index(field.hir_id);
                if self.check_field(hir_id, use_ctxt, adt, &variant.fields[index]) {
                    failed_fields.push((field.ident.name, field.ident.span, true));
                }
            }
            self.emit_unreachable_field_error(failed_fields, adt, None, qpath.span());
        }

        intravisit::walk_pat(self, pat);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////
/// Type privacy visitor, checks types for privacy and reports violations.
/// Both explicitly written types and inferred types of expressions and patterns are checked.
/// Checks are performed on "semantic" types regardless of names and their hygiene.
////////////////////////////////////////////////////////////////////////////////////////////

struct TypePrivacyVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    module_def_id: LocalModDefId,
    maybe_typeck_results: Option<&'tcx ty::TypeckResults<'tcx>>,
    span: Span,
}

impl<'tcx> TypePrivacyVisitor<'tcx> {
    fn item_is_accessible(&self, did: DefId) -> bool {
        self.tcx.visibility(did).is_accessible_from(self.module_def_id, self.tcx)
    }

    // Take node-id of an expression or pattern and check its type for privacy.
    fn check_expr_pat_type(&mut self, id: hir::HirId, span: Span) -> bool {
        self.span = span;
        let typeck_results = self
            .maybe_typeck_results
            .unwrap_or_else(|| span_bug!(span, "`hir::Expr` or `hir::Pat` outside of a body"));
        let result: ControlFlow<()> = try {
            self.visit(typeck_results.node_type(id))?;
            self.visit(typeck_results.node_args(id))?;
            if let Some(adjustments) = typeck_results.adjustments().get(id) {
                adjustments.iter().try_for_each(|adjustment| self.visit(adjustment.target))?;
            }
        };
        result.is_break()
    }

    fn check_def_id(&mut self, def_id: DefId, kind: &str, descr: &dyn fmt::Display) -> bool {
        let is_error = !self.item_is_accessible(def_id);
        if is_error {
            self.tcx.dcx().emit_err(ItemIsPrivate { span: self.span, kind, descr: descr.into() });
        }
        is_error
    }
}

impl<'tcx> rustc_ty_utils::sig_types::SpannedTypeVisitor<'tcx> for TypePrivacyVisitor<'tcx> {
    type Result = ControlFlow<()>;
    fn visit(&mut self, span: Span, value: impl TypeVisitable<TyCtxt<'tcx>>) -> Self::Result {
        self.span = span;
        value.visit_with(&mut self.skeleton())
    }
}

impl<'tcx> Visitor<'tcx> for TypePrivacyVisitor<'tcx> {
    fn visit_nested_body(&mut self, body_id: hir::BodyId) {
        let old_maybe_typeck_results =
            self.maybe_typeck_results.replace(self.tcx.typeck_body(body_id));
        self.visit_body(self.tcx.hir_body(body_id));
        self.maybe_typeck_results = old_maybe_typeck_results;
    }

    fn visit_ty(&mut self, hir_ty: &'tcx hir::Ty<'tcx, AmbigArg>) {
        self.span = hir_ty.span;
        if self
            .visit(
                self.maybe_typeck_results
                    .unwrap_or_else(|| span_bug!(hir_ty.span, "`hir::Ty` outside of a body"))
                    .node_type(hir_ty.hir_id),
            )
            .is_break()
        {
            return;
        }

        intravisit::walk_ty(self, hir_ty);
    }

    fn visit_infer(
        &mut self,
        inf_id: rustc_hir::HirId,
        inf_span: Span,
        _kind: InferKind<'tcx>,
    ) -> Self::Result {
        self.span = inf_span;
        if let Some(ty) = self
            .maybe_typeck_results
            .unwrap_or_else(|| span_bug!(inf_span, "Inference variable outside of a body"))
            .node_type_opt(inf_id)
        {
            if self.visit(ty).is_break() {
                return;
            }
        } else {
            // FIXME: check types of const infers here.
        }

        self.visit_id(inf_id)
    }

    // Check types of expressions
    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        if self.check_expr_pat_type(expr.hir_id, expr.span) {
            // Do not check nested expressions if the error already happened.
            return;
        }
        match expr.kind {
            hir::ExprKind::Assign(_, rhs, _) | hir::ExprKind::Match(rhs, ..) => {
                // Do not report duplicate errors for `x = y` and `match x { ... }`.
                if self.check_expr_pat_type(rhs.hir_id, rhs.span) {
                    return;
                }
            }
            hir::ExprKind::MethodCall(segment, ..) => {
                // Method calls have to be checked specially.
                self.span = segment.ident.span;
                let typeck_results = self
                    .maybe_typeck_results
                    .unwrap_or_else(|| span_bug!(self.span, "`hir::Expr` outside of a body"));
                if let Some(def_id) = typeck_results.type_dependent_def_id(expr.hir_id) {
                    if self.visit(self.tcx.type_of(def_id).instantiate_identity()).is_break() {
                        return;
                    }
                } else {
                    self.tcx
                        .dcx()
                        .span_delayed_bug(expr.span, "no type-dependent def for method call");
                }
            }
            _ => {}
        }

        intravisit::walk_expr(self, expr);
    }

    // Prohibit access to associated items with insufficient nominal visibility.
    //
    // Additionally, until better reachability analysis for macros 2.0 is available,
    // we prohibit access to private statics from other crates, this allows to give
    // more code internal visibility at link time. (Access to private functions
    // is already prohibited by type privacy for function types.)
    fn visit_qpath(&mut self, qpath: &'tcx hir::QPath<'tcx>, id: hir::HirId, span: Span) {
        let def = match qpath {
            hir::QPath::Resolved(_, path) => match path.res {
                Res::Def(kind, def_id) => Some((kind, def_id)),
                _ => None,
            },
            hir::QPath::TypeRelative(..) | hir::QPath::LangItem(..) => {
                match self.maybe_typeck_results {
                    Some(typeck_results) => typeck_results.type_dependent_def(id),
                    // FIXME: Check type-relative associated types in signatures.
                    None => None,
                }
            }
        };
        let def = def.filter(|(kind, _)| {
            matches!(
                kind,
                DefKind::AssocFn | DefKind::AssocConst | DefKind::AssocTy | DefKind::Static { .. }
            )
        });
        if let Some((kind, def_id)) = def {
            let is_local_static =
                if let DefKind::Static { .. } = kind { def_id.is_local() } else { false };
            if !self.item_is_accessible(def_id) && !is_local_static {
                let name = match *qpath {
                    hir::QPath::LangItem(it, ..) => {
                        self.tcx.lang_items().get(it).map(|did| self.tcx.def_path_str(did))
                    }
                    hir::QPath::Resolved(_, path) => Some(self.tcx.def_path_str(path.res.def_id())),
                    hir::QPath::TypeRelative(_, segment) => Some(segment.ident.to_string()),
                };
                let kind = self.tcx.def_descr(def_id);
                let sess = self.tcx.sess;
                let _ = match name {
                    Some(name) => {
                        sess.dcx().emit_err(ItemIsPrivate { span, kind, descr: (&name).into() })
                    }
                    None => sess.dcx().emit_err(UnnamedItemIsPrivate { span, kind }),
                };
                return;
            }
        }

        intravisit::walk_qpath(self, qpath, id);
    }

    // Check types of patterns.
    fn visit_pat(&mut self, pattern: &'tcx hir::Pat<'tcx>) {
        if self.check_expr_pat_type(pattern.hir_id, pattern.span) {
            // Do not check nested patterns if the error already happened.
            return;
        }

        intravisit::walk_pat(self, pattern);
    }

    fn visit_local(&mut self, local: &'tcx hir::LetStmt<'tcx>) {
        if let Some(init) = local.init {
            if self.check_expr_pat_type(init.hir_id, init.span) {
                // Do not report duplicate errors for `let x = y`.
                return;
            }
        }

        intravisit::walk_local(self, local);
    }
}

impl<'tcx> DefIdVisitor<'tcx> for TypePrivacyVisitor<'tcx> {
    type Result = ControlFlow<()>;
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
    fn visit_def_id(
        &mut self,
        def_id: DefId,
        kind: &str,
        descr: &dyn fmt::Display,
    ) -> Self::Result {
        if self.check_def_id(def_id, kind, descr) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
/// SearchInterfaceForPrivateItemsVisitor traverses an item's interface and
/// finds any private components in it.
/// PrivateItemsInPublicInterfacesVisitor ensures there are no private types
/// and traits in public interfaces.
///////////////////////////////////////////////////////////////////////////////

struct SearchInterfaceForPrivateItemsVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    item_def_id: LocalDefId,
    /// The visitor checks that each component type is at least this visible.
    required_visibility: ty::Visibility,
    required_effective_vis: Option<EffectiveVisibility>,
    in_assoc_ty: bool,
    in_primary_interface: bool,
    skip_assoc_tys: bool,
}

impl SearchInterfaceForPrivateItemsVisitor<'_> {
    fn generics(&mut self) -> &mut Self {
        self.in_primary_interface = true;
        for param in &self.tcx.generics_of(self.item_def_id).own_params {
            match param.kind {
                GenericParamDefKind::Lifetime => {}
                GenericParamDefKind::Type { has_default, .. } => {
                    if has_default {
                        let _ = self.visit(self.tcx.type_of(param.def_id).instantiate_identity());
                    }
                }
                // FIXME(generic_const_exprs): May want to look inside const here
                GenericParamDefKind::Const { .. } => {
                    let _ = self.visit(self.tcx.type_of(param.def_id).instantiate_identity());
                }
            }
        }
        self
    }

    fn predicates(&mut self) -> &mut Self {
        self.in_primary_interface = false;
        // N.B., we use `explicit_predicates_of` and not `predicates_of`
        // because we don't want to report privacy errors due to where
        // clauses that the compiler inferred. We only want to
        // consider the ones that the user wrote. This is important
        // for the inferred outlives rules; see
        // `tests/ui/rfc-2093-infer-outlives/privacy.rs`.
        let _ = self.visit_predicates(self.tcx.explicit_predicates_of(self.item_def_id));
        self
    }

    fn bounds(&mut self) -> &mut Self {
        self.in_primary_interface = false;
        let _ = self.visit_clauses(self.tcx.explicit_item_bounds(self.item_def_id).skip_binder());
        self
    }

    fn ty(&mut self) -> &mut Self {
        self.in_primary_interface = true;
        let _ = self.visit(self.tcx.type_of(self.item_def_id).instantiate_identity());
        self
    }

    fn trait_ref(&mut self) -> &mut Self {
        self.in_primary_interface = true;
        if let Some(trait_ref) = self.tcx.impl_trait_ref(self.item_def_id) {
            let _ = self.visit_trait(trait_ref.instantiate_identity());
        }
        self
    }

    fn check_def_id(&mut self, def_id: DefId, kind: &str, descr: &dyn fmt::Display) -> bool {
        if self.leaks_private_dep(def_id) {
            self.tcx.emit_node_span_lint(
                lint::builtin::EXPORTED_PRIVATE_DEPENDENCIES,
                self.tcx.local_def_id_to_hir_id(self.item_def_id),
                self.tcx.def_span(self.item_def_id.to_def_id()),
                FromPrivateDependencyInPublicInterface {
                    kind,
                    descr: descr.into(),
                    krate: self.tcx.crate_name(def_id.krate),
                },
            );
        }

        let Some(local_def_id) = def_id.as_local() else {
            return false;
        };

        let vis = self.tcx.local_visibility(local_def_id);
        let span = self.tcx.def_span(self.item_def_id.to_def_id());
        let vis_span = self.tcx.def_span(def_id);
        if self.in_assoc_ty && !vis.is_at_least(self.required_visibility, self.tcx) {
            let vis_descr = match vis {
                ty::Visibility::Public => "public",
                ty::Visibility::Restricted(vis_def_id) => {
                    if vis_def_id
                        == self.tcx.parent_module_from_def_id(local_def_id).to_local_def_id()
                    {
                        "private"
                    } else if vis_def_id.is_top_level_module() {
                        "crate-private"
                    } else {
                        "restricted"
                    }
                }
            };

            self.tcx.dcx().emit_err(InPublicInterface {
                span,
                vis_descr,
                kind,
                descr: descr.into(),
                vis_span,
            });
            return false;
        }

        let Some(effective_vis) = self.required_effective_vis else {
            return false;
        };

        let reachable_at_vis = *effective_vis.at_level(Level::Reachable);

        if !vis.is_at_least(reachable_at_vis, self.tcx) {
            let lint = if self.in_primary_interface {
                lint::builtin::PRIVATE_INTERFACES
            } else {
                lint::builtin::PRIVATE_BOUNDS
            };
            self.tcx.emit_node_span_lint(
                lint,
                self.tcx.local_def_id_to_hir_id(self.item_def_id),
                span,
                PrivateInterfacesOrBoundsLint {
                    item_span: span,
                    item_kind: self.tcx.def_descr(self.item_def_id.to_def_id()),
                    item_descr: (&LazyDefPathStr {
                        def_id: self.item_def_id.to_def_id(),
                        tcx: self.tcx,
                    })
                        .into(),
                    item_vis_descr: &reachable_at_vis.to_string(self.item_def_id, self.tcx),
                    ty_span: vis_span,
                    ty_kind: kind,
                    ty_descr: descr.into(),
                    ty_vis_descr: &vis.to_string(local_def_id, self.tcx),
                },
            );
        }

        false
    }

    /// An item is 'leaked' from a private dependency if all
    /// of the following are true:
    /// 1. It's contained within a public type
    /// 2. It comes from a private crate
    fn leaks_private_dep(&self, item_id: DefId) -> bool {
        let ret = self.required_visibility.is_public() && self.tcx.is_private_dep(item_id.krate);

        debug!("leaks_private_dep(item_id={:?})={}", item_id, ret);
        ret
    }
}

impl<'tcx> DefIdVisitor<'tcx> for SearchInterfaceForPrivateItemsVisitor<'tcx> {
    type Result = ControlFlow<()>;
    fn skip_assoc_tys(&self) -> bool {
        self.skip_assoc_tys
    }
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
    fn visit_def_id(
        &mut self,
        def_id: DefId,
        kind: &str,
        descr: &dyn fmt::Display,
    ) -> Self::Result {
        if self.check_def_id(def_id, kind, descr) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

struct PrivateItemsInPublicInterfacesChecker<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    effective_visibilities: &'a EffectiveVisibilities,
}

impl<'tcx> PrivateItemsInPublicInterfacesChecker<'_, 'tcx> {
    fn check(
        &self,
        def_id: LocalDefId,
        required_visibility: ty::Visibility,
        required_effective_vis: Option<EffectiveVisibility>,
    ) -> SearchInterfaceForPrivateItemsVisitor<'tcx> {
        SearchInterfaceForPrivateItemsVisitor {
            tcx: self.tcx,
            item_def_id: def_id,
            required_visibility,
            required_effective_vis,
            in_assoc_ty: false,
            in_primary_interface: true,
            skip_assoc_tys: false,
        }
    }

    fn check_unnameable(&self, def_id: LocalDefId, effective_vis: Option<EffectiveVisibility>) {
        let Some(effective_vis) = effective_vis else {
            return;
        };

        let reexported_at_vis = effective_vis.at_level(Level::Reexported);
        let reachable_at_vis = effective_vis.at_level(Level::Reachable);

        if reachable_at_vis.is_public() && reexported_at_vis != reachable_at_vis {
            let hir_id = self.tcx.local_def_id_to_hir_id(def_id);
            let span = self.tcx.def_span(def_id.to_def_id());
            self.tcx.emit_node_span_lint(
                lint::builtin::UNNAMEABLE_TYPES,
                hir_id,
                span,
                UnnameableTypesLint {
                    span,
                    kind: self.tcx.def_descr(def_id.to_def_id()),
                    descr: (&LazyDefPathStr { def_id: def_id.to_def_id(), tcx: self.tcx }).into(),
                    reachable_vis: &reachable_at_vis.to_string(def_id, self.tcx),
                    reexported_vis: &reexported_at_vis.to_string(def_id, self.tcx),
                },
            );
        }
    }

    fn check_assoc_item(
        &self,
        def_id: LocalDefId,
        assoc_item_kind: AssocItemKind,
        vis: ty::Visibility,
        effective_vis: Option<EffectiveVisibility>,
    ) {
        let mut check = self.check(def_id, vis, effective_vis);

        let (check_ty, is_assoc_ty) = match assoc_item_kind {
            AssocItemKind::Const | AssocItemKind::Fn { .. } => (true, false),
            AssocItemKind::Type => (self.tcx.defaultness(def_id).has_value(), true),
        };

        check.in_assoc_ty = is_assoc_ty;
        check.generics().predicates();
        if check_ty {
            check.ty();
        }
    }

    fn get(&self, def_id: LocalDefId) -> Option<EffectiveVisibility> {
        self.effective_visibilities.effective_vis(def_id).copied()
    }

    fn check_item(&mut self, id: ItemId) {
        let tcx = self.tcx;
        let def_id = id.owner_id.def_id;
        let item_visibility = tcx.local_visibility(def_id);
        let effective_vis = self.get(def_id);
        let def_kind = tcx.def_kind(def_id);

        match def_kind {
            DefKind::Const | DefKind::Static { .. } | DefKind::Fn | DefKind::TyAlias => {
                if let DefKind::TyAlias = def_kind {
                    self.check_unnameable(def_id, effective_vis);
                }
                self.check(def_id, item_visibility, effective_vis).generics().predicates().ty();
            }
            DefKind::OpaqueTy => {
                // `ty()` for opaque types is the underlying type,
                // it's not a part of interface, so we skip it.
                self.check(def_id, item_visibility, effective_vis).generics().bounds();
            }
            DefKind::Trait => {
                let item = tcx.hir_item(id);
                if let hir::ItemKind::Trait(.., trait_item_refs) = item.kind {
                    self.check_unnameable(item.owner_id.def_id, effective_vis);

                    self.check(item.owner_id.def_id, item_visibility, effective_vis)
                        .generics()
                        .predicates();

                    for trait_item_ref in trait_item_refs {
                        self.check_assoc_item(
                            trait_item_ref.id.owner_id.def_id,
                            trait_item_ref.kind,
                            item_visibility,
                            effective_vis,
                        );

                        if let AssocItemKind::Type = trait_item_ref.kind {
                            self.check(
                                trait_item_ref.id.owner_id.def_id,
                                item_visibility,
                                effective_vis,
                            )
                            .bounds();
                        }
                    }
                }
            }
            DefKind::TraitAlias => {
                self.check(def_id, item_visibility, effective_vis).generics().predicates();
            }
            DefKind::Enum => {
                let item = tcx.hir_item(id);
                if let hir::ItemKind::Enum(_, ref def, _) = item.kind {
                    self.check_unnameable(item.owner_id.def_id, effective_vis);

                    self.check(item.owner_id.def_id, item_visibility, effective_vis)
                        .generics()
                        .predicates();

                    for variant in def.variants {
                        for field in variant.data.fields() {
                            self.check(field.def_id, item_visibility, effective_vis).ty();
                        }
                    }
                }
            }
            // Subitems of foreign modules have their own publicity.
            DefKind::ForeignMod => {
                let item = tcx.hir_item(id);
                if let hir::ItemKind::ForeignMod { items, .. } = item.kind {
                    for foreign_item in items {
                        let foreign_item = tcx.hir_foreign_item(foreign_item.id);

                        let ev = self.get(foreign_item.owner_id.def_id);
                        let vis = tcx.local_visibility(foreign_item.owner_id.def_id);

                        if let ForeignItemKind::Type = foreign_item.kind {
                            self.check_unnameable(foreign_item.owner_id.def_id, ev);
                        }

                        self.check(foreign_item.owner_id.def_id, vis, ev)
                            .generics()
                            .predicates()
                            .ty();
                    }
                }
            }
            // Subitems of structs and unions have their own publicity.
            DefKind::Struct | DefKind::Union => {
                let item = tcx.hir_item(id);
                if let hir::ItemKind::Struct(_, ref struct_def, _)
                | hir::ItemKind::Union(_, ref struct_def, _) = item.kind
                {
                    self.check_unnameable(item.owner_id.def_id, effective_vis);
                    self.check(item.owner_id.def_id, item_visibility, effective_vis)
                        .generics()
                        .predicates();

                    for field in struct_def.fields() {
                        let field_visibility = tcx.local_visibility(field.def_id);
                        let field_ev = self.get(field.def_id);

                        self.check(
                            field.def_id,
                            min(item_visibility, field_visibility, tcx),
                            field_ev,
                        )
                        .ty();
                    }
                }
            }
            // An inherent impl is public when its type is public
            // Subitems of inherent impls have their own publicity.
            // A trait impl is public when both its type and its trait are public
            // Subitems of trait impls have inherited publicity.
            DefKind::Impl { .. } => {
                let item = tcx.hir_item(id);
                if let hir::ItemKind::Impl(impl_) = item.kind {
                    let impl_vis = ty::Visibility::of_impl::<false>(
                        item.owner_id.def_id,
                        tcx,
                        &Default::default(),
                    );

                    // We are using the non-shallow version here, unlike when building the
                    // effective visisibilities table to avoid large number of false positives.
                    // For example in
                    //
                    // impl From<Priv> for Pub {
                    //     fn from(_: Priv) -> Pub {...}
                    // }
                    //
                    // lints shouldn't be emitted even if `from` effective visibility
                    // is larger than `Priv` nominal visibility and if `Priv` can leak
                    // in some scenarios due to type inference.
                    let impl_ev = EffectiveVisibility::of_impl::<false>(
                        item.owner_id.def_id,
                        tcx,
                        self.effective_visibilities,
                    );

                    let mut check = self.check(item.owner_id.def_id, impl_vis, Some(impl_ev));
                    // Generics and predicates of trait impls are intentionally not checked
                    // for private components (#90586).
                    if impl_.of_trait.is_none() {
                        check.generics().predicates();
                    }
                    // Skip checking private components in associated types, due to lack of full
                    // normalization they produce very ridiculous false positives.
                    // FIXME: Remove this when full normalization is implemented.
                    check.skip_assoc_tys = true;
                    check.ty().trait_ref();

                    for impl_item_ref in impl_.items {
                        let impl_item_vis = if impl_.of_trait.is_none() {
                            min(
                                tcx.local_visibility(impl_item_ref.id.owner_id.def_id),
                                impl_vis,
                                tcx,
                            )
                        } else {
                            impl_vis
                        };

                        let impl_item_ev = if impl_.of_trait.is_none() {
                            self.get(impl_item_ref.id.owner_id.def_id)
                                .map(|ev| ev.min(impl_ev, self.tcx))
                        } else {
                            Some(impl_ev)
                        };

                        self.check_assoc_item(
                            impl_item_ref.id.owner_id.def_id,
                            impl_item_ref.kind,
                            impl_item_vis,
                            impl_item_ev,
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        effective_visibilities,
        check_private_in_public,
        check_mod_privacy,
        ..*providers
    };
}

fn check_mod_privacy(tcx: TyCtxt<'_>, module_def_id: LocalModDefId) {
    // Check privacy of names not checked in previous compilation stages.
    let mut visitor = NamePrivacyVisitor { tcx, maybe_typeck_results: None };
    tcx.hir_visit_item_likes_in_module(module_def_id, &mut visitor);

    // Check privacy of explicitly written types and traits as well as
    // inferred types of expressions and patterns.
    let span = tcx.def_span(module_def_id);
    let mut visitor = TypePrivacyVisitor { tcx, module_def_id, maybe_typeck_results: None, span };

    let module = tcx.hir_module_items(module_def_id);
    for def_id in module.definitions() {
        let _ = rustc_ty_utils::sig_types::walk_types(tcx, def_id, &mut visitor);

        if let Some(body_id) = tcx.hir_maybe_body_owned_by(def_id) {
            visitor.visit_nested_body(body_id.id());
        }
    }

    for id in module.free_items() {
        if let ItemKind::Impl(i) = tcx.hir_item(id).kind {
            if let Some(item) = i.of_trait {
                let trait_ref = tcx.impl_trait_ref(id.owner_id.def_id).unwrap();
                let trait_ref = trait_ref.instantiate_identity();
                visitor.span = item.path.span;
                let _ = visitor.visit_def_id(
                    trait_ref.def_id,
                    "trait",
                    &trait_ref.print_only_trait_path(),
                );
            }
        }
    }
}

fn effective_visibilities(tcx: TyCtxt<'_>, (): ()) -> &EffectiveVisibilities {
    // Build up a set of all exported items in the AST. This is a set of all
    // items which are reachable from external crates based on visibility.
    let mut visitor = EmbargoVisitor {
        tcx,
        effective_visibilities: tcx.resolutions(()).effective_visibilities.clone(),
        macro_reachable: Default::default(),
        changed: false,
    };

    visitor.effective_visibilities.check_invariants(tcx);

    // HACK(jynelson): trying to infer the type of `impl Trait` breaks `async-std` (and
    // `pub async fn` in general). Since rustdoc never needs to do codegen and doesn't
    // care about link-time reachability, keep them unreachable (issue #75100).
    let impl_trait_pass = !tcx.sess.opts.actually_rustdoc;
    if impl_trait_pass {
        // Underlying types of `impl Trait`s are marked as reachable unconditionally,
        // so this pass doesn't need to be a part of the fixed point iteration below.
        let krate = tcx.hir_crate_items(());
        for id in krate.opaques() {
            let opaque = tcx.hir_node_by_def_id(id).expect_opaque_ty();
            let should_visit = match opaque.origin {
                hir::OpaqueTyOrigin::FnReturn {
                    parent,
                    in_trait_or_impl: Some(hir::RpitContext::Trait),
                }
                | hir::OpaqueTyOrigin::AsyncFn {
                    parent,
                    in_trait_or_impl: Some(hir::RpitContext::Trait),
                } => match tcx.hir_node_by_def_id(parent).expect_trait_item().expect_fn().1 {
                    hir::TraitFn::Required(_) => false,
                    hir::TraitFn::Provided(..) => true,
                },

                // Always visit RPITs in functions that have definitions,
                // and all TAITs.
                hir::OpaqueTyOrigin::FnReturn {
                    in_trait_or_impl: None | Some(hir::RpitContext::TraitImpl),
                    ..
                }
                | hir::OpaqueTyOrigin::AsyncFn {
                    in_trait_or_impl: None | Some(hir::RpitContext::TraitImpl),
                    ..
                }
                | hir::OpaqueTyOrigin::TyAlias { .. } => true,
            };
            if should_visit {
                // FIXME: This is some serious pessimization intended to workaround deficiencies
                // in the reachability pass (`middle/reachable.rs`). Types are marked as link-time
                // reachable if they are returned via `impl Trait`, even from private functions.
                let pub_ev = EffectiveVisibility::from_vis(ty::Visibility::Public);
                visitor
                    .reach_through_impl_trait(opaque.def_id, pub_ev)
                    .generics()
                    .predicates()
                    .ty();
            }
        }

        visitor.changed = false;
    }

    loop {
        tcx.hir_visit_all_item_likes_in_crate(&mut visitor);
        if visitor.changed {
            visitor.changed = false;
        } else {
            break;
        }
    }
    visitor.effective_visibilities.check_invariants(tcx);

    let mut check_visitor =
        TestReachabilityVisitor { tcx, effective_visibilities: &visitor.effective_visibilities };
    check_visitor.effective_visibility_diagnostic(CRATE_DEF_ID);
    tcx.hir_visit_all_item_likes_in_crate(&mut check_visitor);

    tcx.arena.alloc(visitor.effective_visibilities)
}

fn check_private_in_public(tcx: TyCtxt<'_>, (): ()) {
    let effective_visibilities = tcx.effective_visibilities(());
    // Check for private types in public interfaces.
    let mut checker = PrivateItemsInPublicInterfacesChecker { tcx, effective_visibilities };

    for id in tcx.hir_free_items() {
        checker.check_item(id);
    }
}
