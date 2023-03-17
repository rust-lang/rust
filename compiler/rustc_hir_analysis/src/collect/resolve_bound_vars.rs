//! Resolution of early vs late bound lifetimes.
//!
//! Name resolution for lifetimes is performed on the AST and embedded into HIR. From this
//! information, typechecking needs to transform the lifetime parameters into bound lifetimes.
//! Lifetimes can be early-bound or late-bound. Construction of typechecking terms needs to visit
//! the types in HIR to identify late-bound lifetimes and assign their Debruijn indices. This file
//! is also responsible for assigning their semantics to implicit lifetimes in trait objects.

use rustc_ast::walk_list;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap, FxIndexSet};
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{GenericArg, GenericParam, GenericParamKind, HirIdMap, LifetimeName, Node};
use rustc_middle::bug;
use rustc_middle::hir::nested_filter;
use rustc_middle::middle::resolve_bound_vars::*;
use rustc_middle::ty::{self, TyCtxt, TypeSuperVisitable, TypeVisitor};
use rustc_session::lint;
use rustc_span::def_id::DefId;
use rustc_span::symbol::{sym, Ident};
use rustc_span::Span;
use std::fmt;

use crate::errors;

trait RegionExt {
    fn early(param: &GenericParam<'_>) -> (LocalDefId, ResolvedArg);

    fn late(index: u32, param: &GenericParam<'_>) -> (LocalDefId, ResolvedArg);

    fn id(&self) -> Option<DefId>;

    fn shifted(self, amount: u32) -> ResolvedArg;
}

impl RegionExt for ResolvedArg {
    fn early(param: &GenericParam<'_>) -> (LocalDefId, ResolvedArg) {
        debug!("ResolvedArg::early: def_id={:?}", param.def_id);
        (param.def_id, ResolvedArg::EarlyBound(param.def_id.to_def_id()))
    }

    fn late(idx: u32, param: &GenericParam<'_>) -> (LocalDefId, ResolvedArg) {
        let depth = ty::INNERMOST;
        debug!(
            "ResolvedArg::late: idx={:?}, param={:?} depth={:?} def_id={:?}",
            idx, param, depth, param.def_id,
        );
        (param.def_id, ResolvedArg::LateBound(depth, idx, param.def_id.to_def_id()))
    }

    fn id(&self) -> Option<DefId> {
        match *self {
            ResolvedArg::StaticLifetime | ResolvedArg::Error(_) => None,

            ResolvedArg::EarlyBound(id)
            | ResolvedArg::LateBound(_, _, id)
            | ResolvedArg::Free(_, id) => Some(id),
        }
    }

    fn shifted(self, amount: u32) -> ResolvedArg {
        match self {
            ResolvedArg::LateBound(debruijn, idx, id) => {
                ResolvedArg::LateBound(debruijn.shifted_in(amount), idx, id)
            }
            _ => self,
        }
    }
}

/// Maps the id of each bound variable reference to the variable decl
/// that it corresponds to.
///
/// FIXME. This struct gets converted to a `ResolveBoundVars` for
/// actual use. It has the same data, but indexed by `LocalDefId`. This
/// is silly.
#[derive(Debug, Default)]
struct NamedVarMap {
    // maps from every use of a named (not anonymous) bound var to a
    // `ResolvedArg` describing how that variable is bound
    defs: HirIdMap<ResolvedArg>,

    // Maps relevant hir items to the bound vars on them. These include:
    // - function defs
    // - function pointers
    // - closures
    // - trait refs
    // - bound types (like `T` in `for<'a> T<'a>: Foo`)
    late_bound_vars: HirIdMap<Vec<ty::BoundVariableKind>>,
}

struct BoundVarContext<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    map: &'a mut NamedVarMap,
    scope: ScopeRef<'a>,
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
        bound_vars: FxIndexMap<LocalDefId, ResolvedArg>,

        scope_type: BinderScopeType,

        /// The late bound vars for a given item are stored by `HirId` to be
        /// queried later. However, if we enter an elision scope, we have to
        /// later append the elided bound vars to the list and need to know what
        /// to append to.
        hir_id: hir::HirId,

        s: ScopeRef<'a>,

        /// If this binder comes from a where clause, specify how it was created.
        /// This is used to diagnose inaccessible lifetimes in APIT:
        /// ```ignore (illustrative)
        /// fn foo(x: impl for<'a> Trait<'a, Assoc = impl Copy + 'a>) {}
        /// ```
        where_bound_origin: Option<hir::PredicateOrigin>,
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
    /// on them (e.g., due to ambiguity).
    Elision {
        s: ScopeRef<'a>,
    },

    /// Use a specific lifetime (if `Some`) or leave it unset (to be
    /// inferred in a function body or potentially error outside one),
    /// for the default choice of lifetime in a trait object type.
    ObjectLifetimeDefault {
        lifetime: Option<ResolvedArg>,
        s: ScopeRef<'a>,
    },

    /// When we have nested trait refs, we concatenate late bound vars for inner
    /// trait refs from outer ones. But we also need to include any HRTB
    /// lifetimes encountered when identifying the trait that an associated type
    /// is declared on.
    Supertrait {
        bound_vars: Vec<ty::BoundVariableKind>,
        s: ScopeRef<'a>,
    },

    TraitRefBoundary {
        s: ScopeRef<'a>,
    },

    /// Disallows capturing non-lifetime binders from parent scopes.
    ///
    /// This is necessary for something like `for<T> [(); { /* references T */ }]:`,
    /// since we don't do something more correct like replacing any captured
    /// late-bound vars with early-bound params in the const's own generics.
    AnonConstBoundary {
        s: ScopeRef<'a>,
    },

    Root {
        opt_parent_item: Option<LocalDefId>,
    },
}

#[derive(Copy, Clone, Debug)]
enum BinderScopeType {
    /// Any non-concatenating binder scopes.
    Normal,
    /// Within a syntactic trait ref, there may be multiple poly trait refs that
    /// are nested (under the `associated_type_bounds` feature). The binders of
    /// the inner poly trait refs are extended from the outer poly trait refs
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
            Scope::Binder { bound_vars, scope_type, hir_id, where_bound_origin, s: _ } => f
                .debug_struct("Binder")
                .field("bound_vars", bound_vars)
                .field("scope_type", scope_type)
                .field("hir_id", hir_id)
                .field("where_bound_origin", where_bound_origin)
                .field("s", &"..")
                .finish(),
            Scope::Body { id, s: _ } => {
                f.debug_struct("Body").field("id", id).field("s", &"..").finish()
            }
            Scope::Elision { s: _ } => f.debug_struct("Elision").field("s", &"..").finish(),
            Scope::ObjectLifetimeDefault { lifetime, s: _ } => f
                .debug_struct("ObjectLifetimeDefault")
                .field("lifetime", lifetime)
                .field("s", &"..")
                .finish(),
            Scope::Supertrait { bound_vars, s: _ } => f
                .debug_struct("Supertrait")
                .field("bound_vars", bound_vars)
                .field("s", &"..")
                .finish(),
            Scope::TraitRefBoundary { s: _ } => f.debug_struct("TraitRefBoundary").finish(),
            Scope::AnonConstBoundary { s: _ } => f.debug_struct("AnonConstBoundary").finish(),
            Scope::Root { opt_parent_item } => {
                f.debug_struct("Root").field("opt_parent_item", &opt_parent_item).finish()
            }
        }
    }
}

type ScopeRef<'a> = &'a Scope<'a>;

pub(crate) fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers {
        resolve_bound_vars,

        named_variable_map: |tcx, id| tcx.resolve_bound_vars(id).defs.get(&id),
        is_late_bound_map,
        object_lifetime_default,
        late_bound_vars_map: |tcx, id| tcx.resolve_bound_vars(id).late_bound_vars.get(&id),

        ..*providers
    };
}

/// Computes the `ResolveBoundVars` map that contains data for an entire `Item`.
/// You should not read the result of this query directly, but rather use
/// `named_variable_map`, `is_late_bound_map`, etc.
#[instrument(level = "debug", skip(tcx))]
fn resolve_bound_vars(tcx: TyCtxt<'_>, local_def_id: hir::OwnerId) -> ResolveBoundVars {
    let mut named_variable_map =
        NamedVarMap { defs: Default::default(), late_bound_vars: Default::default() };
    let mut visitor = BoundVarContext {
        tcx,
        map: &mut named_variable_map,
        scope: &Scope::Root { opt_parent_item: None },
    };
    match tcx.hir().owner(local_def_id) {
        hir::OwnerNode::Item(item) => visitor.visit_item(item),
        hir::OwnerNode::ForeignItem(item) => visitor.visit_foreign_item(item),
        hir::OwnerNode::TraitItem(item) => {
            let scope =
                Scope::Root { opt_parent_item: Some(tcx.local_parent(item.owner_id.def_id)) };
            visitor.scope = &scope;
            visitor.visit_trait_item(item)
        }
        hir::OwnerNode::ImplItem(item) => {
            let scope =
                Scope::Root { opt_parent_item: Some(tcx.local_parent(item.owner_id.def_id)) };
            visitor.scope = &scope;
            visitor.visit_impl_item(item)
        }
        hir::OwnerNode::Crate(_) => {}
    }

    let mut rl = ResolveBoundVars::default();

    for (hir_id, v) in named_variable_map.defs {
        let map = rl.defs.entry(hir_id.owner).or_default();
        map.insert(hir_id.local_id, v);
    }
    for (hir_id, v) in named_variable_map.late_bound_vars {
        let map = rl.late_bound_vars.entry(hir_id.owner).or_default();
        map.insert(hir_id.local_id, v);
    }

    debug!(?rl.defs);
    debug!(?rl.late_bound_vars);
    rl
}

fn late_arg_as_bound_arg<'tcx>(
    tcx: TyCtxt<'tcx>,
    arg: &ResolvedArg,
    param: &GenericParam<'tcx>,
) -> ty::BoundVariableKind {
    match arg {
        ResolvedArg::LateBound(_, _, def_id) => {
            let name = tcx.hir().name(tcx.hir().local_def_id_to_hir_id(def_id.expect_local()));
            match param.kind {
                GenericParamKind::Lifetime { .. } => {
                    ty::BoundVariableKind::Region(ty::BrNamed(*def_id, name))
                }
                GenericParamKind::Type { .. } => {
                    ty::BoundVariableKind::Ty(ty::BoundTyKind::Param(*def_id, name))
                }
                GenericParamKind::Const { .. } => ty::BoundVariableKind::Const,
            }
        }
        _ => bug!("{:?} is not a late argument", arg),
    }
}

impl<'a, 'tcx> BoundVarContext<'a, 'tcx> {
    /// Returns the binders in scope and the type of `Binder` that should be created for a poly trait ref.
    fn poly_trait_ref_binder_info(&mut self) -> (Vec<ty::BoundVariableKind>, BinderScopeType) {
        let mut scope = self.scope;
        let mut supertrait_bound_vars = vec![];
        loop {
            match scope {
                Scope::Body { .. } | Scope::Root { .. } => {
                    break (vec![], BinderScopeType::Normal);
                }

                Scope::Elision { s, .. }
                | Scope::ObjectLifetimeDefault { s, .. }
                | Scope::AnonConstBoundary { s } => {
                    scope = s;
                }

                Scope::Supertrait { s, bound_vars } => {
                    supertrait_bound_vars = bound_vars.clone();
                    scope = s;
                }

                Scope::TraitRefBoundary { .. } => {
                    // We should only see super trait lifetimes if there is a `Binder` above
                    assert!(supertrait_bound_vars.is_empty());
                    break (vec![], BinderScopeType::Normal);
                }

                Scope::Binder { hir_id, .. } => {
                    // Nested poly trait refs have the binders concatenated
                    let mut full_binders =
                        self.map.late_bound_vars.entry(*hir_id).or_default().clone();
                    full_binders.extend(supertrait_bound_vars.into_iter());
                    break (full_binders, BinderScopeType::Concatenating);
                }
            }
        }
    }

    fn visit_poly_trait_ref_inner(
        &mut self,
        trait_ref: &'tcx hir::PolyTraitRef<'tcx>,
        non_lifetime_binder_allowed: NonLifetimeBinderAllowed,
    ) {
        debug!("visit_poly_trait_ref(trait_ref={:?})", trait_ref);

        let (mut binders, scope_type) = self.poly_trait_ref_binder_info();

        let initial_bound_vars = binders.len() as u32;
        let mut bound_vars: FxIndexMap<LocalDefId, ResolvedArg> = FxIndexMap::default();
        let binders_iter =
            trait_ref.bound_generic_params.iter().enumerate().map(|(late_bound_idx, param)| {
                let pair = ResolvedArg::late(initial_bound_vars + late_bound_idx as u32, param);
                let r = late_arg_as_bound_arg(self.tcx, &pair.1, param);
                bound_vars.insert(pair.0, pair.1);
                r
            });
        binders.extend(binders_iter);

        if let NonLifetimeBinderAllowed::Deny(where_) = non_lifetime_binder_allowed {
            deny_non_region_late_bound(self.tcx, &mut bound_vars, where_);
        }

        debug!(?binders);
        self.record_late_bound_vars(trait_ref.trait_ref.hir_ref_id, binders);

        // Always introduce a scope here, even if this is in a where clause and
        // we introduced the binders around the bounded Ty. In that case, we
        // just reuse the concatenation functionality also present in nested trait
        // refs.
        let scope = Scope::Binder {
            hir_id: trait_ref.trait_ref.hir_ref_id,
            bound_vars,
            s: self.scope,
            scope_type,
            where_bound_origin: None,
        };
        self.with(scope, |this| {
            walk_list!(this, visit_generic_param, trait_ref.bound_generic_params);
            this.visit_trait_ref(&trait_ref.trait_ref);
        });
    }
}

enum NonLifetimeBinderAllowed {
    Deny(&'static str),
    Allow,
}

impl<'a, 'tcx> Visitor<'tcx> for BoundVarContext<'a, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_nested_body(&mut self, body: hir::BodyId) {
        let body = self.tcx.hir().body(body);
        self.with(Scope::Body { id: body.id(), s: self.scope }, |this| {
            this.visit_body(body);
        });
    }

    fn visit_expr(&mut self, e: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Closure(hir::Closure {
            binder, bound_generic_params, fn_decl, ..
        }) = e.kind
        {
            if let &hir::ClosureBinder::For { span: for_sp, .. } = binder {
                fn span_of_infer(ty: &hir::Ty<'_>) -> Option<Span> {
                    struct V(Option<Span>);

                    impl<'v> Visitor<'v> for V {
                        fn visit_ty(&mut self, t: &'v hir::Ty<'v>) {
                            match t.kind {
                                _ if self.0.is_some() => (),
                                hir::TyKind::Infer => {
                                    self.0 = Some(t.span);
                                }
                                _ => intravisit::walk_ty(self, t),
                            }
                        }
                    }

                    let mut v = V(None);
                    v.visit_ty(ty);
                    v.0
                }

                let infer_in_rt_sp = match fn_decl.output {
                    hir::FnRetTy::DefaultReturn(sp) => Some(sp),
                    hir::FnRetTy::Return(ty) => span_of_infer(ty),
                };

                let infer_spans = fn_decl
                    .inputs
                    .into_iter()
                    .filter_map(span_of_infer)
                    .chain(infer_in_rt_sp)
                    .collect::<Vec<_>>();

                if !infer_spans.is_empty() {
                    self.tcx.sess
                        .struct_span_err(
                            infer_spans,
                            "implicit types in closure signatures are forbidden when `for<...>` is present",
                        )
                        .span_label(for_sp, "`for<...>` is here")
                        .emit();
                }
            }

            let (mut bound_vars, binders): (FxIndexMap<LocalDefId, ResolvedArg>, Vec<_>) =
                bound_generic_params
                    .iter()
                    .enumerate()
                    .map(|(late_bound_idx, param)| {
                        let pair = ResolvedArg::late(late_bound_idx as u32, param);
                        let r = late_arg_as_bound_arg(self.tcx, &pair.1, param);
                        (pair, r)
                    })
                    .unzip();

            deny_non_region_late_bound(self.tcx, &mut bound_vars, "closures");

            self.record_late_bound_vars(e.hir_id, binders);
            let scope = Scope::Binder {
                hir_id: e.hir_id,
                bound_vars,
                s: self.scope,
                scope_type: BinderScopeType::Normal,
                where_bound_origin: None,
            };

            self.with(scope, |this| {
                // a closure has no bounds, so everything
                // contained within is scoped within its binder.
                intravisit::walk_expr(this, e)
            });
        } else {
            intravisit::walk_expr(self, e)
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn visit_item(&mut self, item: &'tcx hir::Item<'tcx>) {
        match &item.kind {
            hir::ItemKind::Impl(hir::Impl { of_trait, .. }) => {
                if let Some(of_trait) = of_trait {
                    self.record_late_bound_vars(of_trait.hir_ref_id, Vec::default());
                }
            }
            _ => {}
        }
        match item.kind {
            hir::ItemKind::Fn(_, generics, _) => {
                self.visit_early_late(item.hir_id(), generics, |this| {
                    intravisit::walk_item(this, item);
                });
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
                self.with(Scope::Elision { s: self.scope }, |this| {
                    intravisit::walk_item(this, item)
                });
            }
            hir::ItemKind::OpaqueTy(hir::OpaqueTy {
                origin: hir::OpaqueTyOrigin::TyAlias, ..
            }) => {
                // Opaque types are visited when we visit the
                // `TyKind::OpaqueDef`, so that they have the lifetimes from
                // their parent opaque_ty in scope.
                //
                // The core idea here is that since OpaqueTys are generated with the impl Trait as
                // their owner, we can keep going until we find the Item that owns that. We then
                // conservatively add all resolved lifetimes. Otherwise we run into problems in
                // cases like `type Foo<'a> = impl Bar<As = impl Baz + 'a>`.
                let parent_item = self.tcx.hir().get_parent_item(item.hir_id());
                let resolved_lifetimes: &ResolveBoundVars =
                    self.tcx.resolve_bound_vars(parent_item);
                // We need to add *all* deps, since opaque tys may want them from *us*
                for (&owner, defs) in resolved_lifetimes.defs.iter() {
                    defs.iter().for_each(|(&local_id, region)| {
                        self.map.defs.insert(hir::HirId { owner, local_id }, *region);
                    });
                }
                for (&owner, late_bound_vars) in resolved_lifetimes.late_bound_vars.iter() {
                    late_bound_vars.iter().for_each(|(&local_id, late_bound_vars)| {
                        self.record_late_bound_vars(
                            hir::HirId { owner, local_id },
                            late_bound_vars.clone(),
                        );
                    });
                }
            }
            hir::ItemKind::OpaqueTy(hir::OpaqueTy {
                origin: hir::OpaqueTyOrigin::FnReturn(parent) | hir::OpaqueTyOrigin::AsyncFn(parent),
                generics,
                ..
            }) => {
                // We want to start our early-bound indices at the end of the parent scope,
                // not including any parent `impl Trait`s.
                let mut bound_vars = FxIndexMap::default();
                debug!(?generics.params);
                for param in generics.params {
                    let (def_id, reg) = ResolvedArg::early(&param);
                    bound_vars.insert(def_id, reg);
                }

                let scope = Scope::Root { opt_parent_item: Some(parent) };
                self.with(scope, |this| {
                    let scope = Scope::Binder {
                        hir_id: item.hir_id(),
                        bound_vars,
                        s: this.scope,
                        scope_type: BinderScopeType::Normal,
                        where_bound_origin: None,
                    };
                    this.with(scope, |this| {
                        let scope = Scope::TraitRefBoundary { s: this.scope };
                        this.with(scope, |this| intravisit::walk_item(this, item))
                    });
                })
            }
            hir::ItemKind::TyAlias(_, generics)
            | hir::ItemKind::Enum(_, generics)
            | hir::ItemKind::Struct(_, generics)
            | hir::ItemKind::Union(_, generics)
            | hir::ItemKind::Trait(_, _, generics, ..)
            | hir::ItemKind::TraitAlias(generics, ..)
            | hir::ItemKind::Impl(&hir::Impl { generics, .. }) => {
                // These kinds of items have only early-bound lifetime parameters.
                let bound_vars = generics.params.iter().map(ResolvedArg::early).collect();
                self.record_late_bound_vars(item.hir_id(), vec![]);
                let scope = Scope::Binder {
                    hir_id: item.hir_id(),
                    bound_vars,
                    scope_type: BinderScopeType::Normal,
                    s: self.scope,
                    where_bound_origin: None,
                };
                self.with(scope, |this| {
                    let scope = Scope::TraitRefBoundary { s: this.scope };
                    this.with(scope, |this| {
                        intravisit::walk_item(this, item);
                    });
                });
            }
        }
    }

    fn visit_foreign_item(&mut self, item: &'tcx hir::ForeignItem<'tcx>) {
        match item.kind {
            hir::ForeignItemKind::Fn(_, _, generics) => {
                self.visit_early_late(item.hir_id(), generics, |this| {
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

    #[instrument(level = "debug", skip(self))]
    fn visit_ty(&mut self, ty: &'tcx hir::Ty<'tcx>) {
        match ty.kind {
            hir::TyKind::BareFn(c) => {
                let (mut bound_vars, binders): (FxIndexMap<LocalDefId, ResolvedArg>, Vec<_>) = c
                    .generic_params
                    .iter()
                    .enumerate()
                    .map(|(late_bound_idx, param)| {
                        let pair = ResolvedArg::late(late_bound_idx as u32, param);
                        let r = late_arg_as_bound_arg(self.tcx, &pair.1, param);
                        (pair, r)
                    })
                    .unzip();

                deny_non_region_late_bound(self.tcx, &mut bound_vars, "function pointer types");

                self.record_late_bound_vars(ty.hir_id, binders);
                let scope = Scope::Binder {
                    hir_id: ty.hir_id,
                    bound_vars,
                    s: self.scope,
                    scope_type: BinderScopeType::Normal,
                    where_bound_origin: None,
                };
                self.with(scope, |this| {
                    // a bare fn has no bounds, so everything
                    // contained within is scoped within its binder.
                    intravisit::walk_ty(this, ty);
                });
            }
            hir::TyKind::TraitObject(bounds, lifetime, _) => {
                debug!(?bounds, ?lifetime, "TraitObject");
                let scope = Scope::TraitRefBoundary { s: self.scope };
                self.with(scope, |this| {
                    for bound in bounds {
                        this.visit_poly_trait_ref_inner(
                            bound,
                            NonLifetimeBinderAllowed::Deny("trait object types"),
                        );
                    }
                });
                match lifetime.res {
                    LifetimeName::ImplicitObjectLifetimeDefault => {
                        // If the user does not write *anything*, we
                        // use the object lifetime defaulting
                        // rules. So e.g., `Box<dyn Debug>` becomes
                        // `Box<dyn Debug + 'static>`.
                        self.resolve_object_lifetime_default(lifetime)
                    }
                    LifetimeName::Infer => {
                        // If the user writes `'_`, we use the *ordinary* elision
                        // rules. So the `'_` in e.g., `Box<dyn Debug + '_>` will be
                        // resolved the same as the `'_` in `&'_ Foo`.
                        //
                        // cc #48468
                    }
                    LifetimeName::Param(..) | LifetimeName::Static => {
                        // If the user wrote an explicit name, use that.
                        self.visit_lifetime(lifetime);
                    }
                    LifetimeName::Error => {}
                }
            }
            hir::TyKind::Ref(lifetime_ref, ref mt) => {
                self.visit_lifetime(lifetime_ref);
                let scope = Scope::ObjectLifetimeDefault {
                    lifetime: self.map.defs.get(&lifetime_ref.hir_id).cloned(),
                    s: self.scope,
                };
                self.with(scope, |this| this.visit_ty(&mt.ty));
            }
            hir::TyKind::OpaqueDef(item_id, lifetimes, _in_trait) => {
                // Resolve the lifetimes in the bounds to the lifetime defs in the generics.
                // `fn foo<'a>() -> impl MyTrait<'a> { ... }` desugars to
                // `type MyAnonTy<'b> = impl MyTrait<'b>;`
                //                 ^                  ^ this gets resolved in the scope of
                //                                      the opaque_ty generics
                let opaque_ty = self.tcx.hir().item(item_id);
                match &opaque_ty.kind {
                    hir::ItemKind::OpaqueTy(hir::OpaqueTy {
                        origin: hir::OpaqueTyOrigin::TyAlias,
                        ..
                    }) => {
                        intravisit::walk_ty(self, ty);

                        // Elided lifetimes are not allowed in non-return
                        // position impl Trait
                        let scope = Scope::TraitRefBoundary { s: self.scope };
                        self.with(scope, |this| {
                            let scope = Scope::Elision { s: this.scope };
                            this.with(scope, |this| {
                                intravisit::walk_item(this, opaque_ty);
                            })
                        });

                        return;
                    }
                    hir::ItemKind::OpaqueTy(hir::OpaqueTy {
                        origin: hir::OpaqueTyOrigin::FnReturn(..) | hir::OpaqueTyOrigin::AsyncFn(..),
                        ..
                    }) => {}
                    i => bug!("`impl Trait` pointed to non-opaque type?? {:#?}", i),
                };

                // Resolve the lifetimes that are applied to the opaque type.
                // These are resolved in the current scope.
                // `fn foo<'a>() -> impl MyTrait<'a> { ... }` desugars to
                // `fn foo<'a>() -> MyAnonTy<'a> { ... }`
                //          ^                 ^this gets resolved in the current scope
                for lifetime in lifetimes {
                    let hir::GenericArg::Lifetime(lifetime) = lifetime else {
                        continue
                    };
                    self.visit_lifetime(lifetime);

                    // Check for predicates like `impl for<'a> Trait<impl OtherTrait<'a>>`
                    // and ban them. Type variables instantiated inside binders aren't
                    // well-supported at the moment, so this doesn't work.
                    // In the future, this should be fixed and this error should be removed.
                    let def = self.map.defs.get(&lifetime.hir_id).cloned();
                    let Some(ResolvedArg::LateBound(_, _, def_id)) = def else {
                        continue
                    };
                    let Some(def_id) = def_id.as_local() else {
                        continue
                    };
                    let hir_id = self.tcx.hir().local_def_id_to_hir_id(def_id);
                    // Ensure that the parent of the def is an item, not HRTB
                    let parent_id = self.tcx.hir().parent_id(hir_id);
                    if !parent_id.is_owner() {
                        struct_span_err!(
                            self.tcx.sess,
                            lifetime.ident.span,
                            E0657,
                            "`impl Trait` can only capture lifetimes bound at the fn or impl level"
                        )
                        .emit();
                        self.uninsert_lifetime_on_error(lifetime, def.unwrap());
                    }
                    if let hir::Node::Item(hir::Item {
                        kind: hir::ItemKind::OpaqueTy { .. }, ..
                    }) = self.tcx.hir().get(parent_id)
                    {
                        let mut err = self.tcx.sess.struct_span_err(
                            lifetime.ident.span,
                            "higher kinded lifetime bounds on nested opaque types are not supported yet",
                        );
                        err.span_note(self.tcx.def_span(def_id), "lifetime declared here");
                        err.emit();
                        self.uninsert_lifetime_on_error(lifetime, def.unwrap());
                    }
                }
            }
            _ => intravisit::walk_ty(self, ty),
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem<'tcx>) {
        use self::hir::TraitItemKind::*;
        match trait_item.kind {
            Fn(_, _) => {
                self.visit_early_late(trait_item.hir_id(), &trait_item.generics, |this| {
                    intravisit::walk_trait_item(this, trait_item)
                });
            }
            Type(bounds, ty) => {
                let generics = &trait_item.generics;
                let bound_vars = generics.params.iter().map(ResolvedArg::early).collect();
                self.record_late_bound_vars(trait_item.hir_id(), vec![]);
                let scope = Scope::Binder {
                    hir_id: trait_item.hir_id(),
                    bound_vars,
                    s: self.scope,
                    scope_type: BinderScopeType::Normal,
                    where_bound_origin: None,
                };
                self.with(scope, |this| {
                    let scope = Scope::TraitRefBoundary { s: this.scope };
                    this.with(scope, |this| {
                        this.visit_generics(generics);
                        for bound in bounds {
                            this.visit_param_bound(bound);
                        }
                        if let Some(ty) = ty {
                            this.visit_ty(ty);
                        }
                    })
                });
            }
            Const(_, _) => {
                // Only methods and types support generics.
                assert!(trait_item.generics.params.is_empty());
                intravisit::walk_trait_item(self, trait_item);
            }
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem<'tcx>) {
        use self::hir::ImplItemKind::*;
        match impl_item.kind {
            Fn(..) => self.visit_early_late(impl_item.hir_id(), &impl_item.generics, |this| {
                intravisit::walk_impl_item(this, impl_item)
            }),
            Type(ty) => {
                let generics = &impl_item.generics;
                let bound_vars: FxIndexMap<LocalDefId, ResolvedArg> =
                    generics.params.iter().map(ResolvedArg::early).collect();
                self.record_late_bound_vars(impl_item.hir_id(), vec![]);
                let scope = Scope::Binder {
                    hir_id: impl_item.hir_id(),
                    bound_vars,
                    s: self.scope,
                    scope_type: BinderScopeType::Normal,
                    where_bound_origin: None,
                };
                self.with(scope, |this| {
                    let scope = Scope::TraitRefBoundary { s: this.scope };
                    this.with(scope, |this| {
                        this.visit_generics(generics);
                        this.visit_ty(ty);
                    })
                });
            }
            Const(_, _) => {
                // Only methods and types support generics.
                assert!(impl_item.generics.params.is_empty());
                intravisit::walk_impl_item(self, impl_item);
            }
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn visit_lifetime(&mut self, lifetime_ref: &'tcx hir::Lifetime) {
        match lifetime_ref.res {
            hir::LifetimeName::Static => {
                self.insert_lifetime(lifetime_ref, ResolvedArg::StaticLifetime)
            }
            hir::LifetimeName::Param(param_def_id) => {
                self.resolve_lifetime_ref(param_def_id, lifetime_ref)
            }
            // If we've already reported an error, just ignore `lifetime_ref`.
            hir::LifetimeName::Error => {}
            // Those will be resolved by typechecking.
            hir::LifetimeName::ImplicitObjectLifetimeDefault | hir::LifetimeName::Infer => {}
        }
    }

    fn visit_path(&mut self, path: &hir::Path<'tcx>, hir_id: hir::HirId) {
        for (i, segment) in path.segments.iter().enumerate() {
            let depth = path.segments.len() - i - 1;
            if let Some(args) = segment.args {
                self.visit_segment_args(path.res, depth, args);
            }
        }
        if let Res::Def(DefKind::TyParam | DefKind::ConstParam, param_def_id) = path.res {
            self.resolve_type_ref(param_def_id.expect_local(), hir_id);
        }
    }

    fn visit_fn(
        &mut self,
        fk: intravisit::FnKind<'tcx>,
        fd: &'tcx hir::FnDecl<'tcx>,
        body_id: hir::BodyId,
        _: Span,
        _: LocalDefId,
    ) {
        let output = match fd.output {
            hir::FnRetTy::DefaultReturn(_) => None,
            hir::FnRetTy::Return(ty) => Some(ty),
        };
        self.visit_fn_like_elision(&fd.inputs, output, matches!(fk, intravisit::FnKind::Closure));
        intravisit::walk_fn_kind(self, fk);
        self.visit_nested_body(body_id)
    }

    fn visit_generics(&mut self, generics: &'tcx hir::Generics<'tcx>) {
        let scope = Scope::TraitRefBoundary { s: self.scope };
        self.with(scope, |this| {
            for param in generics.params {
                match param.kind {
                    GenericParamKind::Lifetime { .. } => {}
                    GenericParamKind::Type { default, .. } => {
                        if let Some(ty) = default {
                            this.visit_ty(ty);
                        }
                    }
                    GenericParamKind::Const { ty, default } => {
                        this.visit_ty(ty);
                        if let Some(default) = default {
                            this.visit_body(this.tcx.hir().body(default.body));
                        }
                    }
                }
            }
            for predicate in generics.predicates {
                match predicate {
                    &hir::WherePredicate::BoundPredicate(hir::WhereBoundPredicate {
                        hir_id,
                        bounded_ty,
                        bounds,
                        bound_generic_params,
                        origin,
                        ..
                    }) => {
                        let (bound_vars, binders): (FxIndexMap<LocalDefId, ResolvedArg>, Vec<_>) =
                            bound_generic_params
                                .iter()
                                .enumerate()
                                .map(|(late_bound_idx, param)| {
                                    let pair = ResolvedArg::late(late_bound_idx as u32, param);
                                    let r = late_arg_as_bound_arg(this.tcx, &pair.1, param);
                                    (pair, r)
                                })
                                .unzip();
                        this.record_late_bound_vars(hir_id, binders.clone());
                        // Even if there are no lifetimes defined here, we still wrap it in a binder
                        // scope. If there happens to be a nested poly trait ref (an error), that
                        // will be `Concatenating` anyways, so we don't have to worry about the depth
                        // being wrong.
                        let scope = Scope::Binder {
                            hir_id,
                            bound_vars,
                            s: this.scope,
                            scope_type: BinderScopeType::Normal,
                            where_bound_origin: Some(origin),
                        };
                        this.with(scope, |this| {
                            this.visit_ty(&bounded_ty);
                            walk_list!(this, visit_param_bound, bounds);
                        })
                    }
                    &hir::WherePredicate::RegionPredicate(hir::WhereRegionPredicate {
                        lifetime,
                        bounds,
                        ..
                    }) => {
                        this.visit_lifetime(lifetime);
                        walk_list!(this, visit_param_bound, bounds);

                        if lifetime.res != hir::LifetimeName::Static {
                            for bound in bounds {
                                let hir::GenericBound::Outlives(lt) = bound else {
                                    continue;
                                };
                                if lt.res != hir::LifetimeName::Static {
                                    continue;
                                }
                                this.insert_lifetime(lt, ResolvedArg::StaticLifetime);
                                this.tcx.struct_span_lint_hir(
                                    lint::builtin::UNUSED_LIFETIMES,
                                    lifetime.hir_id,
                                    lifetime.ident.span,
                                    format!(
                                        "unnecessary lifetime parameter `{}`",
                                        lifetime.ident
                                    ),
                                    |lint| {
                                        let help = &format!(
                                            "you can use the `'static` lifetime directly, in place of `{}`",
                                            lifetime.ident,
                                        );
                                        lint.help(help)
                                    },
                                );
                            }
                        }
                    }
                    &hir::WherePredicate::EqPredicate(hir::WhereEqPredicate {
                        lhs_ty,
                        rhs_ty,
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

                self.record_late_bound_vars(*hir_id, binders);
                let scope = Scope::Binder {
                    hir_id: *hir_id,
                    bound_vars: FxIndexMap::default(),
                    s: self.scope,
                    scope_type,
                    where_bound_origin: None,
                };
                self.with(scope, |this| {
                    intravisit::walk_param_bound(this, bound);
                });
            }
            _ => intravisit::walk_param_bound(self, bound),
        }
    }

    fn visit_poly_trait_ref(&mut self, trait_ref: &'tcx hir::PolyTraitRef<'tcx>) {
        self.visit_poly_trait_ref_inner(trait_ref, NonLifetimeBinderAllowed::Allow);
    }

    fn visit_anon_const(&mut self, c: &'tcx hir::AnonConst) {
        self.with(Scope::AnonConstBoundary { s: self.scope }, |this| {
            intravisit::walk_anon_const(this, c);
        });
    }
}

fn object_lifetime_default(tcx: TyCtxt<'_>, param_def_id: DefId) -> ObjectLifetimeDefault {
    debug_assert_eq!(tcx.def_kind(param_def_id), DefKind::TyParam);
    let param_def_id = param_def_id.expect_local();
    let hir::Node::GenericParam(param) = tcx.hir().get_by_def_id(param_def_id) else {
        bug!("expected GenericParam for object_lifetime_default");
    };
    match param.source {
        hir::GenericParamSource::Generics => {
            let parent_def_id = tcx.local_parent(param_def_id);
            let generics = tcx.hir().get_generics(parent_def_id).unwrap();
            let param_hir_id = tcx.local_def_id_to_hir_id(param_def_id);
            let param = generics.params.iter().find(|p| p.hir_id == param_hir_id).unwrap();

            // Scan the bounds and where-clauses on parameters to extract bounds
            // of the form `T:'a` so as to determine the `ObjectLifetimeDefault`
            // for each type parameter.
            match param.kind {
                GenericParamKind::Type { .. } => {
                    let mut set = Set1::Empty;

                    // Look for `type: ...` where clauses.
                    for bound in generics.bounds_for_param(param_def_id) {
                        // Ignore `for<'a> type: ...` as they can change what
                        // lifetimes mean (although we could "just" handle it).
                        if !bound.bound_generic_params.is_empty() {
                            continue;
                        }

                        for bound in bound.bounds {
                            if let hir::GenericBound::Outlives(lifetime) = bound {
                                set.insert(lifetime.res);
                            }
                        }
                    }

                    match set {
                        Set1::Empty => ObjectLifetimeDefault::Empty,
                        Set1::One(hir::LifetimeName::Static) => ObjectLifetimeDefault::Static,
                        Set1::One(hir::LifetimeName::Param(param_def_id)) => {
                            ObjectLifetimeDefault::Param(param_def_id.to_def_id())
                        }
                        _ => ObjectLifetimeDefault::Ambiguous,
                    }
                }
                _ => {
                    bug!("object_lifetime_default_raw must only be called on a type parameter")
                }
            }
        }
        hir::GenericParamSource::Binder => ObjectLifetimeDefault::Empty,
    }
}

impl<'a, 'tcx> BoundVarContext<'a, 'tcx> {
    fn with<F>(&mut self, wrap_scope: Scope<'_>, f: F)
    where
        F: for<'b> FnOnce(&mut BoundVarContext<'b, 'tcx>),
    {
        let BoundVarContext { tcx, map, .. } = self;
        let mut this = BoundVarContext { tcx: *tcx, map, scope: &wrap_scope };
        let span = debug_span!("scope", scope = ?TruncatedScopeDebug(&this.scope));
        {
            let _enter = span.enter();
            f(&mut this);
        }
    }

    fn record_late_bound_vars(&mut self, hir_id: hir::HirId, binder: Vec<ty::BoundVariableKind>) {
        if let Some(old) = self.map.late_bound_vars.insert(hir_id, binder) {
            bug!(
                "overwrote bound vars for {hir_id:?}:\nold={old:?}\nnew={:?}",
                self.map.late_bound_vars[&hir_id]
            )
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
        hir_id: hir::HirId,
        generics: &'tcx hir::Generics<'tcx>,
        walk: F,
    ) where
        F: for<'b, 'c> FnOnce(&'b mut BoundVarContext<'c, 'tcx>),
    {
        let mut named_late_bound_vars = 0;
        let bound_vars: FxIndexMap<LocalDefId, ResolvedArg> = generics
            .params
            .iter()
            .map(|param| match param.kind {
                GenericParamKind::Lifetime { .. } => {
                    if self.tcx.is_late_bound(param.hir_id) {
                        let late_bound_idx = named_late_bound_vars;
                        named_late_bound_vars += 1;
                        ResolvedArg::late(late_bound_idx, param)
                    } else {
                        ResolvedArg::early(param)
                    }
                }
                GenericParamKind::Type { .. } | GenericParamKind::Const { .. } => {
                    ResolvedArg::early(param)
                }
            })
            .collect();

        let binders: Vec<_> = generics
            .params
            .iter()
            .filter(|param| {
                matches!(param.kind, GenericParamKind::Lifetime { .. })
                    && self.tcx.is_late_bound(param.hir_id)
            })
            .enumerate()
            .map(|(late_bound_idx, param)| {
                let pair = ResolvedArg::late(late_bound_idx as u32, param);
                late_arg_as_bound_arg(self.tcx, &pair.1, param)
            })
            .collect();
        self.record_late_bound_vars(hir_id, binders);
        let scope = Scope::Binder {
            hir_id,
            bound_vars,
            s: self.scope,
            scope_type: BinderScopeType::Normal,
            where_bound_origin: None,
        };
        self.with(scope, walk);
    }

    #[instrument(level = "debug", skip(self))]
    fn resolve_lifetime_ref(
        &mut self,
        region_def_id: LocalDefId,
        lifetime_ref: &'tcx hir::Lifetime,
    ) {
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
                    outermost_body = Some(id);
                    scope = s;
                }

                Scope::Root { opt_parent_item } => {
                    if let Some(parent_item) = opt_parent_item
                        && let parent_generics = self.tcx.generics_of(parent_item)
                        && parent_generics.param_def_id_to_index(self.tcx, region_def_id.to_def_id()).is_some()
                    {
                        break Some(ResolvedArg::EarlyBound(region_def_id.to_def_id()));
                    }
                    break None;
                }

                Scope::Binder { ref bound_vars, scope_type, s, where_bound_origin, .. } => {
                    if let Some(&def) = bound_vars.get(&region_def_id) {
                        break Some(def.shifted(late_depth));
                    }
                    match scope_type {
                        BinderScopeType::Normal => late_depth += 1,
                        BinderScopeType::Concatenating => {}
                    }
                    // Fresh lifetimes in APIT used to be allowed in async fns and forbidden in
                    // regular fns.
                    if let Some(hir::PredicateOrigin::ImplTrait) = where_bound_origin
                        && let hir::LifetimeName::Param(param_id) = lifetime_ref.res
                        && let Some(generics) = self.tcx.hir().get_generics(self.tcx.local_parent(param_id))
                        && let Some(param) = generics.params.iter().find(|p| p.def_id == param_id)
                        && param.is_elided_lifetime()
                        && let hir::IsAsync::NotAsync = self.tcx.asyncness(lifetime_ref.hir_id.owner.def_id)
                        && !self.tcx.features().anonymous_lifetime_in_impl_trait
                    {
                        let mut diag =  rustc_session::parse::feature_err(
                            &self.tcx.sess.parse_sess,
                            sym::anonymous_lifetime_in_impl_trait,
                            lifetime_ref.ident.span,
                            "anonymous lifetimes in `impl Trait` are unstable",
                        );

                        if let Some(generics) =
                            self.tcx.hir().get_generics(lifetime_ref.hir_id.owner.def_id)
                        {
                            let new_param_sugg = if let Some(span) =
                                generics.span_for_lifetime_suggestion()
                            {
                                (span, "'a, ".to_owned())
                            } else {
                                (generics.span, "<'a>".to_owned())
                            };

                            let lifetime_sugg = match lifetime_ref.suggestion_position() {
                                (hir::LifetimeSuggestionPosition::Normal, span) => (span, "'a".to_owned()),
                                (hir::LifetimeSuggestionPosition::Ampersand, span) => (span, "'a ".to_owned()),
                                (hir::LifetimeSuggestionPosition::ElidedPath, span) => (span, "<'a>".to_owned()),
                                (hir::LifetimeSuggestionPosition::ElidedPathArgument, span) => (span, "'a, ".to_owned()),
                                (hir::LifetimeSuggestionPosition::ObjectDefault, span) => (span, "+ 'a".to_owned()),
                            };
                            let suggestions = vec![
                                lifetime_sugg,
                                new_param_sugg,
                            ];

                            diag.span_label(
                                lifetime_ref.ident.span,
                                "expected named lifetime parameter",
                            );
                            diag.multipart_suggestion(
                                "consider introducing a named lifetime parameter",
                                suggestions,
                                rustc_errors::Applicability::MaybeIncorrect,
                            );
                        }

                        diag.emit();
                        return;
                    }
                    scope = s;
                }

                Scope::Elision { s, .. }
                | Scope::ObjectLifetimeDefault { s, .. }
                | Scope::Supertrait { s, .. }
                | Scope::TraitRefBoundary { s, .. }
                | Scope::AnonConstBoundary { s } => {
                    scope = s;
                }
            }
        };

        if let Some(mut def) = result {
            if let ResolvedArg::EarlyBound(..) = def {
                // Do not free early-bound regions, only late-bound ones.
            } else if let Some(body_id) = outermost_body {
                let fn_id = self.tcx.hir().body_owner(body_id);
                match self.tcx.hir().get(fn_id) {
                    Node::Item(hir::Item { owner_id, kind: hir::ItemKind::Fn(..), .. })
                    | Node::TraitItem(hir::TraitItem {
                        owner_id,
                        kind: hir::TraitItemKind::Fn(..),
                        ..
                    })
                    | Node::ImplItem(hir::ImplItem {
                        owner_id,
                        kind: hir::ImplItemKind::Fn(..),
                        ..
                    }) => {
                        def = ResolvedArg::Free(owner_id.to_def_id(), def.id().unwrap());
                    }
                    Node::Expr(hir::Expr { kind: hir::ExprKind::Closure(closure), .. }) => {
                        def = ResolvedArg::Free(closure.def_id.to_def_id(), def.id().unwrap());
                    }
                    _ => {}
                }
            }

            self.insert_lifetime(lifetime_ref, def);
            return;
        }

        // We may fail to resolve higher-ranked lifetimes that are mentioned by APIT.
        // AST-based resolution does not care for impl-trait desugaring, which are the
        // responibility of lowering. This may create a mismatch between the resolution
        // AST found (`region_def_id`) which points to HRTB, and what HIR allows.
        // ```
        // fn foo(x: impl for<'a> Trait<'a, Assoc = impl Copy + 'a>) {}
        // ```
        //
        // In such case, walk back the binders to diagnose it properly.
        let mut scope = self.scope;
        loop {
            match *scope {
                Scope::Binder {
                    where_bound_origin: Some(hir::PredicateOrigin::ImplTrait), ..
                } => {
                    let mut err = self.tcx.sess.struct_span_err(
                        lifetime_ref.ident.span,
                        "`impl Trait` can only mention lifetimes bound at the fn or impl level",
                    );
                    err.span_note(self.tcx.def_span(region_def_id), "lifetime declared here");
                    err.emit();
                    return;
                }
                Scope::Root { .. } => break,
                Scope::Binder { s, .. }
                | Scope::Body { s, .. }
                | Scope::Elision { s, .. }
                | Scope::ObjectLifetimeDefault { s, .. }
                | Scope::Supertrait { s, .. }
                | Scope::TraitRefBoundary { s, .. }
                | Scope::AnonConstBoundary { s } => {
                    scope = s;
                }
            }
        }

        self.tcx.sess.delay_span_bug(
            lifetime_ref.ident.span,
            &format!("Could not resolve {:?} in scope {:#?}", lifetime_ref, self.scope,),
        );
    }

    fn resolve_type_ref(&mut self, param_def_id: LocalDefId, hir_id: hir::HirId) {
        // Walk up the scope chain, tracking the number of fn scopes
        // that we pass through, until we find a lifetime with the
        // given name or we run out of scopes.
        // search.
        let mut late_depth = 0;
        let mut scope = self.scope;
        let mut crossed_anon_const = false;
        let result = loop {
            match *scope {
                Scope::Body { s, .. } => {
                    scope = s;
                }

                Scope::Root { opt_parent_item } => {
                    if let Some(parent_item) = opt_parent_item
                        && let parent_generics = self.tcx.generics_of(parent_item)
                        && parent_generics.param_def_id_to_index(self.tcx, param_def_id.to_def_id()).is_some()
                    {
                        break Some(ResolvedArg::EarlyBound(param_def_id.to_def_id()));
                    }
                    break None;
                }

                Scope::Binder { ref bound_vars, scope_type, s, .. } => {
                    if let Some(&def) = bound_vars.get(&param_def_id) {
                        break Some(def.shifted(late_depth));
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

                Scope::AnonConstBoundary { s } => {
                    crossed_anon_const = true;
                    scope = s;
                }
            }
        };

        if let Some(def) = result {
            if let ResolvedArg::LateBound(..) = def && crossed_anon_const {
                let use_span = self.tcx.hir().span(hir_id);
                let def_span = self.tcx.def_span(param_def_id);
                let guar = match self.tcx.def_kind(param_def_id) {
                    DefKind::ConstParam => {
                        self.tcx.sess.emit_err(errors::CannotCaptureLateBoundInAnonConst::Const {
                            use_span,
                            def_span,
                        })
                    }
                    DefKind::TyParam => {
                        self.tcx.sess.emit_err(errors::CannotCaptureLateBoundInAnonConst::Type {
                            use_span,
                            def_span,
                        })
                    }
                    _ => unreachable!(),
                };
                self.map.defs.insert(hir_id, ResolvedArg::Error(guar));
            } else {
                self.map.defs.insert(hir_id, def);
            }
            return;
        }

        self.tcx.sess.delay_span_bug(
            self.tcx.hir().span(hir_id),
            format!("could not resolve {param_def_id:?}"),
        );
    }

    #[instrument(level = "debug", skip(self))]
    fn visit_segment_args(
        &mut self,
        res: Res,
        depth: usize,
        generic_args: &'tcx hir::GenericArgs<'tcx>,
    ) {
        if generic_args.parenthesized {
            self.visit_fn_like_elision(
                generic_args.inputs(),
                Some(generic_args.bindings[0].ty()),
                false,
            );
            return;
        }

        for arg in generic_args.args {
            if let hir::GenericArg::Lifetime(lt) = arg {
                self.visit_lifetime(lt);
            }
        }

        // Figure out if this is a type/trait segment,
        // which requires object lifetime defaults.
        let type_def_id = match res {
            Res::Def(DefKind::AssocTy, def_id) if depth == 1 => Some(self.tcx.parent(def_id)),
            Res::Def(DefKind::Variant, def_id) if depth == 0 => Some(self.tcx.parent(def_id)),
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

        debug!(?type_def_id);

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
                        Scope::Root { .. } => break false,

                        Scope::Body { .. } => break true,

                        Scope::Binder { s, .. }
                        | Scope::Elision { s, .. }
                        | Scope::ObjectLifetimeDefault { s, .. }
                        | Scope::Supertrait { s, .. }
                        | Scope::TraitRefBoundary { s, .. }
                        | Scope::AnonConstBoundary { s } => {
                            scope = s;
                        }
                    }
                }
            };

            let map = &self.map;
            let generics = self.tcx.generics_of(def_id);

            // `type_def_id` points to an item, so there is nothing to inherit generics from.
            debug_assert_eq!(generics.parent_count, 0);

            let set_to_region = |set: ObjectLifetimeDefault| match set {
                ObjectLifetimeDefault::Empty => {
                    if in_body {
                        None
                    } else {
                        Some(ResolvedArg::StaticLifetime)
                    }
                }
                ObjectLifetimeDefault::Static => Some(ResolvedArg::StaticLifetime),
                ObjectLifetimeDefault::Param(param_def_id) => {
                    // This index can be used with `generic_args` since `parent_count == 0`.
                    let index = generics.param_def_id_to_index[&param_def_id] as usize;
                    generic_args.args.get(index).and_then(|arg| match arg {
                        GenericArg::Lifetime(lt) => map.defs.get(&lt.hir_id).copied(),
                        _ => None,
                    })
                }
                ObjectLifetimeDefault::Ambiguous => None,
            };
            generics
                .params
                .iter()
                .filter_map(|param| {
                    match self.tcx.def_kind(param.def_id) {
                        // Generic consts don't impose any constraints.
                        //
                        // We still store a dummy value here to allow generic parameters
                        // in an arbitrary order.
                        DefKind::ConstParam => Some(ObjectLifetimeDefault::Empty),
                        DefKind::TyParam => Some(self.tcx.object_lifetime_default(param.def_id)),
                        // We may also get a `Trait` or `TraitAlias` because of how generics `Self` parameter
                        // works. Ignore it because it can't have a meaningful lifetime default.
                        DefKind::LifetimeParam | DefKind::Trait | DefKind::TraitAlias => None,
                        dk => bug!("unexpected def_kind {:?}", dk),
                    }
                })
                .map(set_to_region)
                .collect()
        });

        debug!(?object_lifetime_defaults);

        let mut i = 0;
        for arg in generic_args.args {
            match arg {
                GenericArg::Lifetime(_) => {}
                GenericArg::Type(ty) => {
                    if let Some(&lt) = object_lifetime_defaults.get(i) {
                        let scope = Scope::ObjectLifetimeDefault { lifetime: lt, s: self.scope };
                        self.with(scope, |this| this.visit_ty(ty));
                    } else {
                        self.visit_ty(ty);
                    }
                    i += 1;
                }
                GenericArg::Const(ct) => {
                    self.visit_anon_const(&ct.value);
                    i += 1;
                }
                GenericArg::Infer(inf) => {
                    self.visit_id(inf.hir_id);
                    i += 1;
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
                lifetime: if has_lifetime_parameter {
                    None
                } else {
                    Some(ResolvedArg::StaticLifetime)
                },
                s: self.scope,
            };
            if let Some(type_def_id) = type_def_id {
                let bound_vars =
                    BoundVarContext::supertrait_hrtb_vars(self.tcx, type_def_id, binding.ident);
                self.with(scope, |this| {
                    let scope = Scope::Supertrait {
                        bound_vars: bound_vars.unwrap_or_default(),
                        s: this.scope,
                    };
                    this.with(scope, |this| this.visit_assoc_type_binding(binding));
                });
            } else {
                self.with(scope, |this| this.visit_assoc_type_binding(binding));
            }
        }
    }

    /// Returns all the late-bound vars that come into scope from supertrait HRTBs, based on the
    /// associated type name and starting trait.
    /// For example, imagine we have
    /// ```ignore (illustrative)
    /// trait Foo<'a, 'b> {
    ///   type As;
    /// }
    /// trait Bar<'b>: for<'a> Foo<'a, 'b> {}
    /// trait Bar: for<'b> Bar<'b> {}
    /// ```
    /// In this case, if we wanted to the supertrait HRTB lifetimes for `As` on
    /// the starting trait `Bar`, we would return `Some(['b, 'a])`.
    fn supertrait_hrtb_vars(
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
            let Some((def_id, bound_vars)) = stack.pop() else {
                break None;
            };
            // See issue #83753. If someone writes an associated type on a non-trait, just treat it as
            // there being no supertrait HRTBs.
            match tcx.def_kind(def_id) {
                DefKind::Trait | DefKind::TraitAlias | DefKind::Impl { .. } => {}
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
                    ty::PredicateKind::Clause(ty::Clause::Trait(data)) => {
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

    #[instrument(level = "debug", skip(self))]
    fn visit_fn_like_elision(
        &mut self,
        inputs: &'tcx [hir::Ty<'tcx>],
        output: Option<&'tcx hir::Ty<'tcx>>,
        in_closure: bool,
    ) {
        self.with(Scope::Elision { s: self.scope }, |this| {
            for input in inputs {
                this.visit_ty(input);
            }
            if !in_closure && let Some(output) = output {
                this.visit_ty(output);
            }
        });
        if in_closure && let Some(output) = output {
            self.visit_ty(output);
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

                Scope::Root { .. } | Scope::Elision { .. } => break ResolvedArg::StaticLifetime,

                Scope::Body { .. } | Scope::ObjectLifetimeDefault { lifetime: None, .. } => return,

                Scope::ObjectLifetimeDefault { lifetime: Some(l), .. } => break l,

                Scope::Supertrait { s, .. }
                | Scope::TraitRefBoundary { s, .. }
                | Scope::AnonConstBoundary { s } => {
                    scope = s;
                }
            }
        };
        self.insert_lifetime(lifetime_ref, lifetime.shifted(late_depth));
    }

    #[instrument(level = "debug", skip(self))]
    fn insert_lifetime(&mut self, lifetime_ref: &'tcx hir::Lifetime, def: ResolvedArg) {
        debug!(span = ?lifetime_ref.ident.span);
        self.map.defs.insert(lifetime_ref.hir_id, def);
    }

    /// Sometimes we resolve a lifetime, but later find that it is an
    /// error (esp. around impl trait). In that case, we remove the
    /// entry into `map.defs` so as not to confuse later code.
    fn uninsert_lifetime_on_error(
        &mut self,
        lifetime_ref: &'tcx hir::Lifetime,
        bad_def: ResolvedArg,
    ) {
        let old_value = self.map.defs.remove(&lifetime_ref.hir_id);
        assert_eq!(old_value, Some(bad_def));
    }
}

/// Detects late-bound lifetimes and inserts them into
/// `late_bound`.
///
/// A region declared on a fn is **late-bound** if:
/// - it is constrained by an argument type;
/// - it does not appear in a where-clause.
///
/// "Constrained" basically means that it appears in any type but
/// not amongst the inputs to a projection. In other words, `<&'a
/// T as Trait<''b>>::Foo` does not constrain `'a` or `'b`.
fn is_late_bound_map(
    tcx: TyCtxt<'_>,
    owner_id: hir::OwnerId,
) -> Option<&FxIndexSet<hir::ItemLocalId>> {
    let decl = tcx.hir().fn_decl_by_hir_id(owner_id.into())?;
    let generics = tcx.hir().get_generics(owner_id.def_id)?;

    let mut late_bound = FxIndexSet::default();

    let mut constrained_by_input = ConstrainedCollector { regions: Default::default(), tcx };
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

        // appears in the where clauses? early-bound.
        if appears_in_where_clause.regions.contains(&param.def_id) {
            continue;
        }

        // does not appear in the inputs, but appears in the return type? early-bound.
        if !constrained_by_input.regions.contains(&param.def_id)
            && appears_in_output.regions.contains(&param.def_id)
        {
            continue;
        }

        debug!("lifetime {:?} with id {:?} is late-bound", param.name.ident(), param.def_id);

        let inserted = late_bound.insert(param.hir_id.local_id);
        assert!(inserted, "visited lifetime {:?} twice", param.def_id);
    }

    debug!(?late_bound);
    return Some(tcx.arena.alloc(late_bound));

    /// Visits a `ty::Ty` collecting information about what generic parameters are constrained.
    ///
    /// The visitor does not operate on `hir::Ty` so that it can be called on the rhs of a `type Alias<...> = ...;`
    /// which may live in a separate crate so there would not be any hir available. Instead we use the `type_of`
    /// query to obtain a `ty::Ty` which will be present even in cross crate scenarios. It also naturally
    /// handles cycle detection as we go through the query system.
    ///
    /// This is necessary in the first place for the following case:
    /// ```
    /// type Alias<'a, T> = <T as Trait<'a>>::Assoc;
    /// fn foo<'a>(_: Alias<'a, ()>) -> Alias<'a, ()> { ... }
    /// ```
    ///
    /// If we conservatively considered `'a` unconstrained then we could break users who had written code before
    /// we started correctly handling aliases. If we considered `'a` constrained then it would become late bound
    /// causing an error during astconv as the `'a` is not constrained by the input type `<() as Trait<'a>>::Assoc`
    /// but appears in the output type `<() as Trait<'a>>::Assoc`.
    ///
    /// We must therefore "look into" the `Alias` to see whether we should consider `'a` constrained or not.
    ///
    /// See #100508 #85533 #47511 for additional context
    struct ConstrainedCollectorPostAstConv {
        arg_is_constrained: Box<[bool]>,
    }

    use std::ops::ControlFlow;
    use ty::Ty;
    impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for ConstrainedCollectorPostAstConv {
        fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<!> {
            match t.kind() {
                ty::Param(param_ty) => {
                    self.arg_is_constrained[param_ty.index as usize] = true;
                }
                ty::Alias(ty::Projection, _) => return ControlFlow::Continue(()),
                _ => (),
            }
            t.super_visit_with(self)
        }

        fn visit_const(&mut self, _: ty::Const<'tcx>) -> ControlFlow<!> {
            ControlFlow::Continue(())
        }

        fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<!> {
            debug!("r={:?}", r.kind());
            if let ty::RegionKind::ReEarlyBound(region) = r.kind() {
                self.arg_is_constrained[region.index as usize] = true;
            }

            ControlFlow::Continue(())
        }
    }

    struct ConstrainedCollector<'tcx> {
        tcx: TyCtxt<'tcx>,
        regions: FxHashSet<LocalDefId>,
    }

    impl<'v> Visitor<'v> for ConstrainedCollector<'_> {
        fn visit_ty(&mut self, ty: &'v hir::Ty<'v>) {
            match ty.kind {
                hir::TyKind::Path(
                    hir::QPath::Resolved(Some(_), _) | hir::QPath::TypeRelative(..),
                ) => {
                    // ignore lifetimes appearing in associated type
                    // projections, as they are not *constrained*
                    // (defined above)
                }

                hir::TyKind::Path(hir::QPath::Resolved(
                    None,
                    hir::Path { res: Res::Def(DefKind::TyAlias, alias_def), segments, span },
                )) => {
                    // See comments on `ConstrainedCollectorPostAstConv` for why this arm does not just consider
                    // substs to be unconstrained.
                    let generics = self.tcx.generics_of(alias_def);
                    let mut walker = ConstrainedCollectorPostAstConv {
                        arg_is_constrained: vec![false; generics.params.len()].into_boxed_slice(),
                    };
                    walker.visit_ty(self.tcx.type_of(alias_def).subst_identity());

                    match segments.last() {
                        Some(hir::PathSegment { args: Some(args), .. }) => {
                            let tcx = self.tcx;
                            for constrained_arg in
                                args.args.iter().enumerate().flat_map(|(n, arg)| {
                                    match walker.arg_is_constrained.get(n) {
                                        Some(true) => Some(arg),
                                        Some(false) => None,
                                        None => {
                                            tcx.sess.delay_span_bug(
                                                *span,
                                                format!(
                                                    "Incorrect generic arg count for alias {:?}",
                                                    alias_def
                                                ),
                                            );
                                            None
                                        }
                                    }
                                })
                            {
                                self.visit_generic_arg(constrained_arg);
                            }
                        }
                        Some(_) => (),
                        None => bug!("Path with no segments or self type"),
                    }
                }

                hir::TyKind::Path(hir::QPath::Resolved(None, path)) => {
                    // consider only the lifetimes on the final
                    // segment; I am not sure it's even currently
                    // valid to have them elsewhere, but even if it
                    // is, those would be potentially inputs to
                    // projections
                    if let Some(last_segment) = path.segments.last() {
                        self.visit_path_segment(last_segment);
                    }
                }

                _ => {
                    intravisit::walk_ty(self, ty);
                }
            }
        }

        fn visit_lifetime(&mut self, lifetime_ref: &'v hir::Lifetime) {
            if let hir::LifetimeName::Param(def_id) = lifetime_ref.res {
                self.regions.insert(def_id);
            }
        }
    }

    #[derive(Default)]
    struct AllCollector {
        regions: FxHashSet<LocalDefId>,
    }

    impl<'v> Visitor<'v> for AllCollector {
        fn visit_lifetime(&mut self, lifetime_ref: &'v hir::Lifetime) {
            if let hir::LifetimeName::Param(def_id) = lifetime_ref.res {
                self.regions.insert(def_id);
            }
        }
    }
}

pub fn deny_non_region_late_bound(
    tcx: TyCtxt<'_>,
    bound_vars: &mut FxIndexMap<LocalDefId, ResolvedArg>,
    where_: &str,
) {
    let mut first = true;

    for (var, arg) in bound_vars {
        let Node::GenericParam(param) = tcx.hir().get_by_def_id(*var) else {
            bug!();
        };

        let what = match param.kind {
            hir::GenericParamKind::Type { .. } => "type",
            hir::GenericParamKind::Const { .. } => "const",
            hir::GenericParamKind::Lifetime { .. } => continue,
        };

        let mut diag = tcx.sess.struct_span_err(
            param.span,
            format!("late-bound {what} parameter not allowed on {where_}"),
        );

        let guar = if tcx.features().non_lifetime_binders && first {
            diag.emit()
        } else {
            diag.delay_as_bug()
        };

        first = false;
        *arg = ResolvedArg::Error(guar);
    }
}
