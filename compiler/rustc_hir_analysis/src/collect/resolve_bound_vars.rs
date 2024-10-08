//! Resolution of early vs late bound lifetimes.
//!
//! Name resolution for lifetimes is performed on the AST and embedded into HIR. From this
//! information, typechecking needs to transform the lifetime parameters into bound lifetimes.
//! Lifetimes can be early-bound or late-bound. Construction of typechecking terms needs to visit
//! the types in HIR to identify late-bound lifetimes and assign their Debruijn indices. This file
//! is also responsible for assigning their semantics to implicit lifetimes in trait objects.

use core::ops::ControlFlow;
use std::fmt;

use rustc_ast::visit::walk_list;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap, FxIndexSet};
use rustc_data_structures::sorted_map::SortedMap;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{
    GenericArg, GenericParam, GenericParamKind, HirId, ItemLocalMap, LifetimeName, Node,
};
use rustc_macros::extension;
use rustc_middle::hir::nested_filter;
use rustc_middle::middle::resolve_bound_vars::*;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, TyCtxt, TypeSuperVisitable, TypeVisitor};
use rustc_middle::{bug, span_bug};
use rustc_span::Span;
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::symbol::{Ident, sym};
use tracing::{debug, debug_span, instrument};

use crate::errors;

#[extension(trait RegionExt)]
impl ResolvedArg {
    fn early(param: &GenericParam<'_>) -> (LocalDefId, ResolvedArg) {
        debug!("ResolvedArg::early: def_id={:?}", param.def_id);
        (param.def_id, ResolvedArg::EarlyBound(param.def_id))
    }

    fn late(idx: u32, param: &GenericParam<'_>) -> (LocalDefId, ResolvedArg) {
        let depth = ty::INNERMOST;
        debug!(
            "ResolvedArg::late: idx={:?}, param={:?} depth={:?} def_id={:?}",
            idx, param, depth, param.def_id,
        );
        (param.def_id, ResolvedArg::LateBound(depth, idx, param.def_id))
    }

    fn id(&self) -> Option<LocalDefId> {
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
    defs: ItemLocalMap<ResolvedArg>,

    // Maps relevant hir items to the bound vars on them. These include:
    // - function defs
    // - function pointers
    // - closures
    // - trait refs
    // - bound types (like `T` in `for<'a> T<'a>: Foo`)
    late_bound_vars: ItemLocalMap<Vec<ty::BoundVariableKind>>,
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
        hir_id: HirId,

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

    /// Disallows capturing late-bound vars from parent scopes.
    ///
    /// This is necessary for something like `for<T> [(); { /* references T */ }]:`,
    /// since we don't do something more correct like replacing any captured
    /// late-bound vars with early-bound params in the const's own generics.
    LateBoundary {
        s: ScopeRef<'a>,
        what: &'static str,
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
            Scope::LateBoundary { s: _, what } => {
                f.debug_struct("LateBoundary").field("what", what).finish()
            }
            Scope::Root { opt_parent_item } => {
                f.debug_struct("Root").field("opt_parent_item", &opt_parent_item).finish()
            }
        }
    }
}

type ScopeRef<'a> = &'a Scope<'a>;

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers {
        resolve_bound_vars,

        named_variable_map: |tcx, id| &tcx.resolve_bound_vars(id).defs,
        is_late_bound_map,
        object_lifetime_default,
        late_bound_vars_map: |tcx, id| &tcx.resolve_bound_vars(id).late_bound_vars,

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
    match tcx.hir_owner_node(local_def_id) {
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
        hir::OwnerNode::Synthetic => unreachable!(),
    }

    let defs = named_variable_map.defs.into_sorted_stable_ord();
    let late_bound_vars = named_variable_map.late_bound_vars.into_sorted_stable_ord();
    let rl = ResolveBoundVars {
        defs: SortedMap::from_presorted_elements(defs),
        late_bound_vars: SortedMap::from_presorted_elements(late_bound_vars),
    };

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
            let def_id = def_id.to_def_id();
            let name = tcx.item_name(def_id);
            match param.kind {
                GenericParamKind::Lifetime { .. } => {
                    ty::BoundVariableKind::Region(ty::BrNamed(def_id, name))
                }
                GenericParamKind::Type { .. } => {
                    ty::BoundVariableKind::Ty(ty::BoundTyKind::Param(def_id, name))
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

                Scope::ObjectLifetimeDefault { s, .. } | Scope::LateBoundary { s, .. } => {
                    scope = s;
                }

                Scope::Supertrait { s, bound_vars } => {
                    supertrait_bound_vars = bound_vars.clone();
                    scope = s;
                }

                Scope::TraitRefBoundary { .. } => {
                    // We should only see super trait lifetimes if there is a `Binder` above
                    // though this may happen when we call `poly_trait_ref_binder_info` with
                    // an (erroneous, #113423) associated return type bound in an impl header.
                    if !supertrait_bound_vars.is_empty() {
                        self.tcx.dcx().delayed_bug(format!(
                            "found supertrait lifetimes without a binder to append \
                                them to: {supertrait_bound_vars:?}"
                        ));
                    }
                    break (vec![], BinderScopeType::Normal);
                }

                Scope::Binder { hir_id, .. } => {
                    // Nested poly trait refs have the binders concatenated
                    let mut full_binders =
                        self.map.late_bound_vars.entry(hir_id.local_id).or_default().clone();
                    full_binders.extend(supertrait_bound_vars);
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
                    /// Look for `_` anywhere in the signature of a `for<> ||` closure.
                    /// This is currently disallowed.
                    struct FindInferInClosureWithBinder;
                    impl<'v> Visitor<'v> for FindInferInClosureWithBinder {
                        type Result = ControlFlow<Span>;
                        fn visit_ty(&mut self, t: &'v hir::Ty<'v>) -> Self::Result {
                            if matches!(t.kind, hir::TyKind::Infer) {
                                ControlFlow::Break(t.span)
                            } else {
                                intravisit::walk_ty(self, t)
                            }
                        }
                    }
                    FindInferInClosureWithBinder.visit_ty(ty).break_value()
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
                    self.tcx
                        .dcx()
                        .emit_err(errors::ClosureImplicitHrtb { spans: infer_spans, for_sp });
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
    fn visit_opaque_ty(&mut self, opaque: &'tcx rustc_hir::OpaqueTy<'tcx>) {
        // We want to start our early-bound indices at the end of the parent scope,
        // not including any parent `impl Trait`s.
        let mut bound_vars = FxIndexMap::default();
        debug!(?opaque.generics.params);
        for param in opaque.generics.params {
            let (def_id, reg) = ResolvedArg::early(param);
            bound_vars.insert(def_id, reg);
        }

        let hir_id = self.tcx.local_def_id_to_hir_id(opaque.def_id);
        let scope = Scope::Binder {
            hir_id,
            bound_vars,
            s: self.scope,
            scope_type: BinderScopeType::Normal,
            where_bound_origin: None,
        };
        self.with(scope, |this| {
            let scope = Scope::TraitRefBoundary { s: this.scope };
            this.with(scope, |this| intravisit::walk_opaque_ty(this, opaque))
        })
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
            | hir::ItemKind::Static(..)
            | hir::ItemKind::GlobalAsm(..) => {
                // These sorts of items have no lifetime parameters at all.
                intravisit::walk_item(self, item);
            }
            hir::ItemKind::TyAlias(_, generics)
            | hir::ItemKind::Const(_, generics, _)
            | hir::ItemKind::Enum(_, generics)
            | hir::ItemKind::Struct(_, generics)
            | hir::ItemKind::Union(_, generics)
            | hir::ItemKind::Trait(_, _, generics, ..)
            | hir::ItemKind::TraitAlias(generics, ..)
            | hir::ItemKind::Impl(&hir::Impl { generics, .. }) => {
                // These kinds of items have only early-bound lifetime parameters.
                self.visit_early(item.hir_id(), generics, |this| intravisit::walk_item(this, item));
            }
        }
    }

    fn visit_precise_capturing_arg(
        &mut self,
        arg: &'tcx hir::PreciseCapturingArg<'tcx>,
    ) -> Self::Result {
        match *arg {
            hir::PreciseCapturingArg::Lifetime(lt) => match lt.res {
                LifetimeName::Param(def_id) => {
                    self.resolve_lifetime_ref(def_id, lt);
                }
                LifetimeName::Error => {}
                LifetimeName::ImplicitObjectLifetimeDefault
                | LifetimeName::Infer
                | LifetimeName::Static => {
                    self.tcx.dcx().emit_err(errors::BadPreciseCapture {
                        span: lt.ident.span,
                        kind: "lifetime",
                        found: format!("`{}`", lt.ident.name),
                    });
                }
            },
            hir::PreciseCapturingArg::Param(param) => match param.res {
                Res::Def(DefKind::TyParam | DefKind::ConstParam, def_id)
                | Res::SelfTyParam { trait_: def_id } => {
                    self.resolve_type_ref(def_id.expect_local(), param.hir_id);
                }
                Res::SelfTyAlias { alias_to, .. } => {
                    self.tcx.dcx().emit_err(errors::PreciseCaptureSelfAlias {
                        span: param.ident.span,
                        self_span: self.tcx.def_span(alias_to),
                        what: self.tcx.def_descr(alias_to),
                    });
                }
                res => {
                    self.tcx.dcx().span_delayed_bug(
                        param.ident.span,
                        format!("expected type or const param, found {res:?}"),
                    );
                }
            },
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
                    for (bound, _) in bounds {
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
                    lifetime: self.map.defs.get(&lifetime_ref.hir_id.local_id).cloned(),
                    s: self.scope,
                };
                self.with(scope, |this| this.visit_ty(mt.ty));
            }
            hir::TyKind::OpaqueDef(opaque_ty, lifetimes) => {
                self.visit_opaque_ty(opaque_ty);

                // Resolve the lifetimes in the bounds to the lifetime defs in the generics.
                // `fn foo<'a>() -> impl MyTrait<'a> { ... }` desugars to
                // `type MyAnonTy<'b> = impl MyTrait<'b>;`
                //                 ^                  ^ this gets resolved in the scope of
                //                                      the opaque_ty generics

                // Resolve the lifetimes that are applied to the opaque type.
                // These are resolved in the current scope.
                // `fn foo<'a>() -> impl MyTrait<'a> { ... }` desugars to
                // `fn foo<'a>() -> MyAnonTy<'a> { ... }`
                //          ^                 ^this gets resolved in the current scope
                for lifetime in lifetimes {
                    let hir::GenericArg::Lifetime(lifetime) = lifetime else { continue };
                    self.visit_lifetime(lifetime);

                    // Check for predicates like `impl for<'a> Trait<impl OtherTrait<'a>>`
                    // and ban them. Type variables instantiated inside binders aren't
                    // well-supported at the moment, so this doesn't work.
                    // In the future, this should be fixed and this error should be removed.
                    let def = self.map.defs.get(&lifetime.hir_id.local_id).copied();
                    let Some(ResolvedArg::LateBound(_, _, lifetime_def_id)) = def else { continue };
                    let lifetime_hir_id = self.tcx.local_def_id_to_hir_id(lifetime_def_id);

                    let bad_place = match self.tcx.hir_node(self.tcx.parent_hir_id(lifetime_hir_id))
                    {
                        // Opaques do not declare their own lifetimes, so if a lifetime comes from an opaque
                        // it must be a reified late-bound lifetime from a trait goal.
                        hir::Node::OpaqueTy(_) => "higher-ranked lifetime from outer `impl Trait`",
                        // Other items are fine.
                        hir::Node::Item(_) | hir::Node::TraitItem(_) | hir::Node::ImplItem(_) => {
                            continue;
                        }
                        hir::Node::Ty(hir::Ty { kind: hir::TyKind::BareFn(_), .. }) => {
                            "higher-ranked lifetime from function pointer"
                        }
                        hir::Node::Ty(hir::Ty { kind: hir::TyKind::TraitObject(..), .. }) => {
                            "higher-ranked lifetime from `dyn` type"
                        }
                        _ => "higher-ranked lifetime",
                    };

                    let (span, label) = if lifetime.ident.span == self.tcx.def_span(lifetime_def_id)
                    {
                        (opaque_ty.span, Some(opaque_ty.span))
                    } else {
                        (lifetime.ident.span, None)
                    };

                    // Ensure that the parent of the def is an item, not HRTB
                    self.tcx.dcx().emit_err(errors::OpaqueCapturesHigherRankedLifetime {
                        span,
                        label,
                        decl_span: self.tcx.def_span(lifetime_def_id),
                        bad_place,
                    });
                    self.uninsert_lifetime_on_error(lifetime, def.unwrap());
                }
            }
            _ => intravisit::walk_ty(self, ty),
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn visit_pattern_type_pattern(&mut self, p: &'tcx hir::Pat<'tcx>) {
        intravisit::walk_pat(self, p)
    }

    #[instrument(level = "debug", skip(self))]
    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem<'tcx>) {
        use self::hir::TraitItemKind::*;
        match trait_item.kind {
            Fn(_, _) => {
                self.visit_early_late(trait_item.hir_id(), trait_item.generics, |this| {
                    intravisit::walk_trait_item(this, trait_item)
                });
            }
            Type(bounds, ty) => {
                self.visit_early(trait_item.hir_id(), trait_item.generics, |this| {
                    this.visit_generics(trait_item.generics);
                    for bound in bounds {
                        this.visit_param_bound(bound);
                    }
                    if let Some(ty) = ty {
                        this.visit_ty(ty);
                    }
                })
            }
            Const(_, _) => self.visit_early(trait_item.hir_id(), trait_item.generics, |this| {
                intravisit::walk_trait_item(this, trait_item)
            }),
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem<'tcx>) {
        use self::hir::ImplItemKind::*;
        match impl_item.kind {
            Fn(..) => self.visit_early_late(impl_item.hir_id(), impl_item.generics, |this| {
                intravisit::walk_impl_item(this, impl_item)
            }),
            Type(ty) => self.visit_early(impl_item.hir_id(), impl_item.generics, |this| {
                this.visit_generics(impl_item.generics);
                this.visit_ty(ty);
            }),
            Const(_, _) => self.visit_early(impl_item.hir_id(), impl_item.generics, |this| {
                intravisit::walk_impl_item(this, impl_item)
            }),
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

    fn visit_path(&mut self, path: &hir::Path<'tcx>, hir_id: HirId) {
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
        def_id: LocalDefId,
    ) {
        let output = match fd.output {
            hir::FnRetTy::DefaultReturn(_) => None,
            hir::FnRetTy::Return(ty) => Some(ty),
        };
        if let Some(ty) = output
            && let hir::TyKind::InferDelegation(sig_id, _) = ty.kind
        {
            let bound_vars: Vec<_> =
                self.tcx.fn_sig(sig_id).skip_binder().bound_vars().iter().collect();
            let hir_id = self.tcx.local_def_id_to_hir_id(def_id);
            self.map.late_bound_vars.insert(hir_id.local_id, bound_vars);
        }
        self.visit_fn_like_elision(fd.inputs, output, matches!(fk, intravisit::FnKind::Closure));
        intravisit::walk_fn_kind(self, fk);
        self.visit_nested_body(body_id)
    }

    fn visit_generics(&mut self, generics: &'tcx hir::Generics<'tcx>) {
        let scope = Scope::TraitRefBoundary { s: self.scope };
        self.with(scope, |this| {
            walk_list!(this, visit_generic_param, generics.params);
            walk_list!(this, visit_where_predicate, generics.predicates);
        })
    }

    fn visit_where_predicate(&mut self, predicate: &'tcx hir::WherePredicate<'tcx>) {
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
                            let r = late_arg_as_bound_arg(self.tcx, &pair.1, param);
                            (pair, r)
                        })
                        .unzip();

                self.record_late_bound_vars(hir_id, binders);

                // If this is an RTN type in the self type, then append those to the binder.
                self.try_append_return_type_notation_params(hir_id, bounded_ty);

                // Even if there are no lifetimes defined here, we still wrap it in a binder
                // scope. If there happens to be a nested poly trait ref (an error), that
                // will be `Concatenating` anyways, so we don't have to worry about the depth
                // being wrong.
                let scope = Scope::Binder {
                    hir_id,
                    bound_vars,
                    s: self.scope,
                    scope_type: BinderScopeType::Normal,
                    where_bound_origin: Some(origin),
                };
                self.with(scope, |this| {
                    walk_list!(this, visit_generic_param, bound_generic_params);
                    this.visit_ty(bounded_ty);
                    walk_list!(this, visit_param_bound, bounds);
                })
            }
            &hir::WherePredicate::RegionPredicate(hir::WhereRegionPredicate {
                lifetime,
                bounds,
                ..
            }) => {
                self.visit_lifetime(lifetime);
                walk_list!(self, visit_param_bound, bounds);
            }
            &hir::WherePredicate::EqPredicate(hir::WhereEqPredicate { lhs_ty, rhs_ty, .. }) => {
                self.visit_ty(lhs_ty);
                self.visit_ty(rhs_ty);
            }
        }
    }

    fn visit_poly_trait_ref(&mut self, trait_ref: &'tcx hir::PolyTraitRef<'tcx>) {
        self.visit_poly_trait_ref_inner(trait_ref, NonLifetimeBinderAllowed::Allow);
    }

    fn visit_anon_const(&mut self, c: &'tcx hir::AnonConst) {
        self.with(Scope::LateBoundary { s: self.scope, what: "constant" }, |this| {
            intravisit::walk_anon_const(this, c);
        });
    }

    fn visit_generic_param(&mut self, p: &'tcx GenericParam<'tcx>) {
        match p.kind {
            GenericParamKind::Type { .. } | GenericParamKind::Const { .. } => {
                self.resolve_type_ref(p.def_id, p.hir_id);
            }
            GenericParamKind::Lifetime { .. } => {
                // No need to resolve lifetime params, we don't use them for things
                // like implicit `?Sized` or const-param-has-ty predicates.
            }
        }

        match p.kind {
            GenericParamKind::Lifetime { .. } => {}
            GenericParamKind::Type { default, .. } => {
                if let Some(ty) = default {
                    self.visit_ty(ty);
                }
            }
            GenericParamKind::Const { ty, default, .. } => {
                self.visit_ty(ty);
                if let Some(default) = default {
                    self.visit_const_arg(default);
                }
            }
        }
    }
}

fn object_lifetime_default(tcx: TyCtxt<'_>, param_def_id: LocalDefId) -> ObjectLifetimeDefault {
    debug_assert_eq!(tcx.def_kind(param_def_id), DefKind::TyParam);
    let hir::Node::GenericParam(param) = tcx.hir_node_by_def_id(param_def_id) else {
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
        let span = debug_span!("scope", scope = ?TruncatedScopeDebug(this.scope));
        {
            let _enter = span.enter();
            f(&mut this);
        }
    }

    fn record_late_bound_vars(&mut self, hir_id: HirId, binder: Vec<ty::BoundVariableKind>) {
        if let Some(old) = self.map.late_bound_vars.insert(hir_id.local_id, binder) {
            bug!(
                "overwrote bound vars for {hir_id:?}:\nold={old:?}\nnew={:?}",
                self.map.late_bound_vars[&hir_id.local_id]
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
    fn visit_early_late<F>(&mut self, hir_id: HirId, generics: &'tcx hir::Generics<'tcx>, walk: F)
    where
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

    fn visit_early<F>(&mut self, hir_id: HirId, generics: &'tcx hir::Generics<'tcx>, walk: F)
    where
        F: for<'b, 'c> FnOnce(&'b mut BoundVarContext<'c, 'tcx>),
    {
        let bound_vars = generics.params.iter().map(ResolvedArg::early).collect();
        self.record_late_bound_vars(hir_id, vec![]);
        let scope = Scope::Binder {
            hir_id,
            bound_vars,
            s: self.scope,
            scope_type: BinderScopeType::Normal,
            where_bound_origin: None,
        };
        self.with(scope, |this| {
            let scope = Scope::TraitRefBoundary { s: this.scope };
            this.with(scope, walk)
        });
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
        let mut crossed_late_boundary = None;
        let result = loop {
            match *scope {
                Scope::Body { id, s } => {
                    outermost_body = Some(id);
                    scope = s;
                }

                Scope::Root { opt_parent_item } => {
                    if let Some(parent_item) = opt_parent_item
                        && let parent_generics = self.tcx.generics_of(parent_item)
                        && parent_generics
                            .param_def_id_to_index(self.tcx, region_def_id.to_def_id())
                            .is_some()
                    {
                        break Some(ResolvedArg::EarlyBound(region_def_id));
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
                        && let Some(generics) =
                            self.tcx.hir().get_generics(self.tcx.local_parent(param_id))
                        && let Some(param) = generics.params.iter().find(|p| p.def_id == param_id)
                        && param.is_elided_lifetime()
                        && !self.tcx.asyncness(lifetime_ref.hir_id.owner.def_id).is_async()
                        && !self.tcx.features().anonymous_lifetime_in_impl_trait
                    {
                        let mut diag: rustc_errors::Diag<'_> = rustc_session::parse::feature_err(
                            &self.tcx.sess,
                            sym::anonymous_lifetime_in_impl_trait,
                            lifetime_ref.ident.span,
                            "anonymous lifetimes in `impl Trait` are unstable",
                        );

                        if let Some(generics) =
                            self.tcx.hir().get_generics(lifetime_ref.hir_id.owner.def_id)
                        {
                            let new_param_sugg =
                                if let Some(span) = generics.span_for_lifetime_suggestion() {
                                    (span, "'a, ".to_owned())
                                } else {
                                    (generics.span, "<'a>".to_owned())
                                };

                            let lifetime_sugg = lifetime_ref.suggestion("'a");
                            let suggestions = vec![lifetime_sugg, new_param_sugg];

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

                Scope::ObjectLifetimeDefault { s, .. }
                | Scope::Supertrait { s, .. }
                | Scope::TraitRefBoundary { s, .. } => {
                    scope = s;
                }

                Scope::LateBoundary { s, what } => {
                    crossed_late_boundary = Some(what);
                    scope = s;
                }
            }
        };

        if let Some(mut def) = result {
            if let ResolvedArg::EarlyBound(..) = def {
                // Do not free early-bound regions, only late-bound ones.
            } else if let ResolvedArg::LateBound(_, _, param_def_id) = def
                && let Some(what) = crossed_late_boundary
            {
                let use_span = lifetime_ref.ident.span;
                let def_span = self.tcx.def_span(param_def_id);
                let guar = match self.tcx.def_kind(param_def_id) {
                    DefKind::LifetimeParam => {
                        self.tcx.dcx().emit_err(errors::CannotCaptureLateBound::Lifetime {
                            use_span,
                            def_span,
                            what,
                        })
                    }
                    kind => span_bug!(
                        use_span,
                        "did not expect to resolve lifetime to {}",
                        kind.descr(param_def_id.to_def_id())
                    ),
                };
                def = ResolvedArg::Error(guar);
            } else if let Some(body_id) = outermost_body {
                let fn_id = self.tcx.hir().body_owner(body_id);
                match self.tcx.hir_node(fn_id) {
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
                        def = ResolvedArg::Free(owner_id.def_id, def.id().unwrap());
                    }
                    Node::Expr(hir::Expr { kind: hir::ExprKind::Closure(closure), .. }) => {
                        def = ResolvedArg::Free(closure.def_id, def.id().unwrap());
                    }
                    _ => {}
                }
            }

            self.insert_lifetime(lifetime_ref, def);
            return;
        }

        // We may fail to resolve higher-ranked lifetimes that are mentioned by APIT.
        // AST-based resolution does not care for impl-trait desugaring, which are the
        // responsibility of lowering. This may create a mismatch between the resolution
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
                    self.tcx.dcx().emit_err(errors::LateBoundInApit::Lifetime {
                        span: lifetime_ref.ident.span,
                        param_span: self.tcx.def_span(region_def_id),
                    });
                    return;
                }
                Scope::Root { .. } => break,
                Scope::Binder { s, .. }
                | Scope::Body { s, .. }
                | Scope::ObjectLifetimeDefault { s, .. }
                | Scope::Supertrait { s, .. }
                | Scope::TraitRefBoundary { s, .. }
                | Scope::LateBoundary { s, .. } => {
                    scope = s;
                }
            }
        }

        self.tcx.dcx().span_delayed_bug(
            lifetime_ref.ident.span,
            format!("Could not resolve {:?} in scope {:#?}", lifetime_ref, self.scope,),
        );
    }

    fn resolve_type_ref(&mut self, param_def_id: LocalDefId, hir_id: HirId) {
        // Walk up the scope chain, tracking the number of fn scopes
        // that we pass through, until we find a lifetime with the
        // given name or we run out of scopes.
        // search.
        let mut late_depth = 0;
        let mut scope = self.scope;
        let mut crossed_late_boundary = None;

        let result = loop {
            match *scope {
                Scope::Body { s, .. } => {
                    scope = s;
                }

                Scope::Root { opt_parent_item } => {
                    if let Some(parent_item) = opt_parent_item
                        && let parent_generics = self.tcx.generics_of(parent_item)
                        && parent_generics
                            .param_def_id_to_index(self.tcx, param_def_id.to_def_id())
                            .is_some()
                    {
                        break Some(ResolvedArg::EarlyBound(param_def_id));
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

                Scope::ObjectLifetimeDefault { s, .. }
                | Scope::Supertrait { s, .. }
                | Scope::TraitRefBoundary { s, .. } => {
                    scope = s;
                }

                Scope::LateBoundary { s, what } => {
                    crossed_late_boundary = Some(what);
                    scope = s;
                }
            }
        };

        if let Some(def) = result {
            if let ResolvedArg::LateBound(..) = def
                && let Some(what) = crossed_late_boundary
            {
                let use_span = self.tcx.hir().span(hir_id);
                let def_span = self.tcx.def_span(param_def_id);
                let guar = match self.tcx.def_kind(param_def_id) {
                    DefKind::ConstParam => {
                        self.tcx.dcx().emit_err(errors::CannotCaptureLateBound::Const {
                            use_span,
                            def_span,
                            what,
                        })
                    }
                    DefKind::TyParam => {
                        self.tcx.dcx().emit_err(errors::CannotCaptureLateBound::Type {
                            use_span,
                            def_span,
                            what,
                        })
                    }
                    kind => span_bug!(
                        use_span,
                        "did not expect to resolve non-lifetime param to {}",
                        kind.descr(param_def_id.to_def_id())
                    ),
                };
                self.map.defs.insert(hir_id.local_id, ResolvedArg::Error(guar));
            } else {
                self.map.defs.insert(hir_id.local_id, def);
            }
            return;
        }

        // We may fail to resolve higher-ranked ty/const vars that are mentioned by APIT.
        // AST-based resolution does not care for impl-trait desugaring, which are the
        // responsibility of lowering. This may create a mismatch between the resolution
        // AST found (`param_def_id`) which points to HRTB, and what HIR allows.
        // ```
        // fn foo(x: impl for<T> Trait<Assoc = impl Trait2<T>>) {}
        // ```
        //
        // In such case, walk back the binders to diagnose it properly.
        let mut scope = self.scope;
        loop {
            match *scope {
                Scope::Binder {
                    where_bound_origin: Some(hir::PredicateOrigin::ImplTrait), ..
                } => {
                    let guar = self.tcx.dcx().emit_err(match self.tcx.def_kind(param_def_id) {
                        DefKind::TyParam => errors::LateBoundInApit::Type {
                            span: self.tcx.hir().span(hir_id),
                            param_span: self.tcx.def_span(param_def_id),
                        },
                        DefKind::ConstParam => errors::LateBoundInApit::Const {
                            span: self.tcx.hir().span(hir_id),
                            param_span: self.tcx.def_span(param_def_id),
                        },
                        kind => {
                            bug!("unexpected def-kind: {}", kind.descr(param_def_id.to_def_id()))
                        }
                    });
                    self.map.defs.insert(hir_id.local_id, ResolvedArg::Error(guar));
                    return;
                }
                Scope::Root { .. } => break,
                Scope::Binder { s, .. }
                | Scope::Body { s, .. }
                | Scope::ObjectLifetimeDefault { s, .. }
                | Scope::Supertrait { s, .. }
                | Scope::TraitRefBoundary { s, .. }
                | Scope::LateBoundary { s, .. } => {
                    scope = s;
                }
            }
        }

        self.tcx
            .dcx()
            .span_bug(self.tcx.hir().span(hir_id), format!("could not resolve {param_def_id:?}"));
    }

    #[instrument(level = "debug", skip(self))]
    fn visit_segment_args(
        &mut self,
        res: Res,
        depth: usize,
        generic_args: &'tcx hir::GenericArgs<'tcx>,
    ) {
        if let Some((inputs, output)) = generic_args.paren_sugar_inputs_output() {
            self.visit_fn_like_elision(inputs, Some(output), false);
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
                        | Scope::ObjectLifetimeDefault { s, .. }
                        | Scope::Supertrait { s, .. }
                        | Scope::TraitRefBoundary { s, .. }
                        | Scope::LateBoundary { s, .. } => {
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
                        GenericArg::Lifetime(lt) => map.defs.get(&lt.hir_id.local_id).copied(),
                        _ => None,
                    })
                }
                ObjectLifetimeDefault::Ambiguous => None,
            };
            generics
                .own_params
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
                    self.visit_const_arg(ct);
                    i += 1;
                }
                GenericArg::Infer(inf) => {
                    self.visit_id(inf.hir_id);
                    i += 1;
                }
            }
        }

        // Hack: When resolving the type `XX` in an assoc ty binding like
        // `dyn Foo<'b, Item = XX>`, the current object-lifetime default
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
        for constraint in generic_args.constraints {
            let scope = Scope::ObjectLifetimeDefault {
                lifetime: if has_lifetime_parameter {
                    None
                } else {
                    Some(ResolvedArg::StaticLifetime)
                },
                s: self.scope,
            };
            // If the args are parenthesized, then this must be `feature(return_type_notation)`.
            // In that case, introduce a binder over all of the function's early and late bound vars.
            //
            // For example, given
            // ```
            // trait Foo {
            //     async fn x<'r, T>();
            // }
            // ```
            // and a bound that looks like:
            //    `for<'a> T::Trait<'a, x(..): for<'b> Other<'b>>`
            // this is going to expand to something like:
            //    `for<'a> for<'r> <T as Trait<'a>>::x::<'r, T>::{opaque#0}: for<'b> Other<'b>`.
            if constraint.gen_args.parenthesized == hir::GenericArgsParentheses::ReturnTypeNotation
            {
                let bound_vars = if let Some(type_def_id) = type_def_id
                    && self.tcx.def_kind(type_def_id) == DefKind::Trait
                    && let Some((mut bound_vars, assoc_fn)) = BoundVarContext::supertrait_hrtb_vars(
                        self.tcx,
                        type_def_id,
                        constraint.ident,
                        ty::AssocKind::Fn,
                    ) {
                    bound_vars.extend(self.tcx.generics_of(assoc_fn.def_id).own_params.iter().map(
                        |param| match param.kind {
                            ty::GenericParamDefKind::Lifetime => ty::BoundVariableKind::Region(
                                ty::BoundRegionKind::BrNamed(param.def_id, param.name),
                            ),
                            ty::GenericParamDefKind::Type { .. } => ty::BoundVariableKind::Ty(
                                ty::BoundTyKind::Param(param.def_id, param.name),
                            ),
                            ty::GenericParamDefKind::Const { .. } => ty::BoundVariableKind::Const,
                        },
                    ));
                    bound_vars.extend(
                        self.tcx.fn_sig(assoc_fn.def_id).instantiate_identity().bound_vars(),
                    );
                    bound_vars
                } else {
                    self.tcx
                        .dcx()
                        .span_delayed_bug(constraint.ident.span, "bad return type notation here");
                    vec![]
                };
                self.with(scope, |this| {
                    let scope = Scope::Supertrait { bound_vars, s: this.scope };
                    this.with(scope, |this| {
                        let (bound_vars, _) = this.poly_trait_ref_binder_info();
                        this.record_late_bound_vars(constraint.hir_id, bound_vars);
                        this.visit_assoc_item_constraint(constraint)
                    });
                });
            } else if let Some(type_def_id) = type_def_id {
                let bound_vars = BoundVarContext::supertrait_hrtb_vars(
                    self.tcx,
                    type_def_id,
                    constraint.ident,
                    ty::AssocKind::Type,
                )
                .map(|(bound_vars, _)| bound_vars);
                self.with(scope, |this| {
                    let scope = Scope::Supertrait {
                        bound_vars: bound_vars.unwrap_or_default(),
                        s: this.scope,
                    };
                    this.with(scope, |this| this.visit_assoc_item_constraint(constraint));
                });
            } else {
                self.with(scope, |this| this.visit_assoc_item_constraint(constraint));
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
        assoc_kind: ty::AssocKind,
    ) -> Option<(Vec<ty::BoundVariableKind>, &'tcx ty::AssocItem)> {
        let trait_defines_associated_item_named = |trait_def_id: DefId| {
            tcx.associated_items(trait_def_id).find_by_name_and_kind(
                tcx,
                assoc_name,
                assoc_kind,
                trait_def_id,
            )
        };

        use smallvec::{SmallVec, smallvec};
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

            if let Some(assoc_item) = trait_defines_associated_item_named(def_id) {
                break Some((bound_vars.into_iter().collect(), assoc_item));
            }
            let predicates = tcx.explicit_supertraits_containing_assoc_item((def_id, assoc_name));
            let obligations = predicates.iter_identity_copied().filter_map(|(pred, _)| {
                let bound_predicate = pred.kind();
                match bound_predicate.skip_binder() {
                    ty::ClauseKind::Trait(data) => {
                        // The order here needs to match what we would get from
                        // `rustc_middle::ty::predicate::Clause::instantiate_supertrait`
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
        self.with(
            Scope::ObjectLifetimeDefault {
                lifetime: Some(ResolvedArg::StaticLifetime),
                s: self.scope,
            },
            |this| {
                for input in inputs {
                    this.visit_ty(input);
                }
                if !in_closure && let Some(output) = output {
                    this.visit_ty(output);
                }
            },
        );
        if in_closure && let Some(output) = output {
            self.visit_ty(output);
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn resolve_object_lifetime_default(&mut self, lifetime_ref: &'tcx hir::Lifetime) {
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

                Scope::Root { .. } => break ResolvedArg::StaticLifetime,

                Scope::Body { .. } | Scope::ObjectLifetimeDefault { lifetime: None, .. } => return,

                Scope::ObjectLifetimeDefault { lifetime: Some(l), .. } => break l,

                Scope::Supertrait { s, .. }
                | Scope::TraitRefBoundary { s, .. }
                | Scope::LateBoundary { s, .. } => {
                    scope = s;
                }
            }
        };
        self.insert_lifetime(lifetime_ref, lifetime.shifted(late_depth));
    }

    #[instrument(level = "debug", skip(self))]
    fn insert_lifetime(&mut self, lifetime_ref: &'tcx hir::Lifetime, def: ResolvedArg) {
        debug!(span = ?lifetime_ref.ident.span);
        self.map.defs.insert(lifetime_ref.hir_id.local_id, def);
    }

    /// Sometimes we resolve a lifetime, but later find that it is an
    /// error (esp. around impl trait). In that case, we remove the
    /// entry into `map.defs` so as not to confuse later code.
    fn uninsert_lifetime_on_error(
        &mut self,
        lifetime_ref: &'tcx hir::Lifetime,
        bad_def: ResolvedArg,
    ) {
        let old_value = self.map.defs.remove(&lifetime_ref.hir_id.local_id);
        assert_eq!(old_value, Some(bad_def));
    }

    // When we have a return type notation type in a where clause, like
    // `where <T as Trait>::method(..): Send`, we need to introduce new bound
    // vars to the existing where clause's binder, to represent the lifetimes
    // elided by the return-type-notation syntax.
    //
    // For example, given
    // ```
    // trait Foo {
    //     async fn x<'r>();
    // }
    // ```
    // and a bound that looks like:
    //    `for<'a, 'b> <T as Trait<'a>>::x(): Other<'b>`
    // this is going to expand to something like:
    //    `for<'a, 'b, 'r> <T as Trait<'a>>::x::<'r, T>::{opaque#0}: Other<'b>`.
    //
    // We handle this similarly for associated-type-bound style return-type-notation
    // in `visit_segment_args`.
    fn try_append_return_type_notation_params(
        &mut self,
        hir_id: HirId,
        hir_ty: &'tcx hir::Ty<'tcx>,
    ) {
        let hir::TyKind::Path(qpath) = hir_ty.kind else {
            // We only care about path types here. All other self types
            // (including nesting the RTN type in another type) don't do
            // anything.
            return;
        };

        let (mut bound_vars, item_def_id, item_segment) = match qpath {
            // If we have a fully qualified method, then we don't need to do any special lookup.
            hir::QPath::Resolved(_, path)
                if let [.., item_segment] = &path.segments[..]
                    && item_segment.args.is_some_and(|args| {
                        matches!(
                            args.parenthesized,
                            hir::GenericArgsParentheses::ReturnTypeNotation
                        )
                    }) =>
            {
                match path.res {
                    Res::Err => return,
                    Res::Def(DefKind::AssocFn, item_def_id) => (vec![], item_def_id, item_segment),
                    _ => bug!("only expected method resolution for fully qualified RTN"),
                }
            }

            // If we have a type-dependent path, then we do need to do some lookup.
            hir::QPath::TypeRelative(qself, item_segment)
                if item_segment.args.is_some_and(|args| {
                    matches!(args.parenthesized, hir::GenericArgsParentheses::ReturnTypeNotation)
                }) =>
            {
                // First, ignore a qself that isn't a type or `Self` param. Those are the
                // only ones that support `T::Assoc` anyways in HIR lowering.
                let hir::TyKind::Path(hir::QPath::Resolved(None, path)) = qself.kind else {
                    return;
                };
                match path.res {
                    Res::Def(DefKind::TyParam, _) | Res::SelfTyParam { trait_: _ } => {
                        // Get the generics of this type's hir owner. This is *different*
                        // from the generics of the parameter's definition, since we want
                        // to be able to resolve an RTN path on a nested body (e.g. method
                        // inside an impl) using the where clauses on the method.
                        // FIXME(return_type_notation): Think of some better way of doing this.
                        let Some(generics) = self.tcx.hir_owner_node(hir_id.owner).generics()
                        else {
                            return;
                        };

                        // Look for the first bound that contains an associated type that
                        // matches the segment that we're looking for. We ignore any subsequent
                        // bounds since we'll be emitting a hard error in HIR lowering, so this
                        // is purely speculative.
                        let one_bound = generics.predicates.iter().find_map(|predicate| {
                            let hir::WherePredicate::BoundPredicate(predicate) = predicate else {
                                return None;
                            };
                            let hir::TyKind::Path(hir::QPath::Resolved(None, bounded_path)) =
                                predicate.bounded_ty.kind
                            else {
                                return None;
                            };
                            if bounded_path.res != path.res {
                                return None;
                            }
                            predicate.bounds.iter().find_map(|bound| {
                                let hir::GenericBound::Trait(trait_, _) = bound else {
                                    return None;
                                };
                                BoundVarContext::supertrait_hrtb_vars(
                                    self.tcx,
                                    trait_.trait_ref.trait_def_id()?,
                                    item_segment.ident,
                                    ty::AssocKind::Fn,
                                )
                            })
                        });
                        let Some((bound_vars, assoc_item)) = one_bound else {
                            return;
                        };
                        (bound_vars, assoc_item.def_id, item_segment)
                    }
                    // If we have a self type alias (in an impl), try to resolve an
                    // associated item from one of the supertraits of the impl's trait.
                    Res::SelfTyAlias { alias_to: impl_def_id, is_trait_impl: true, .. } => {
                        let hir::ItemKind::Impl(hir::Impl { of_trait: Some(trait_ref), .. }) = self
                            .tcx
                            .hir_node_by_def_id(impl_def_id.expect_local())
                            .expect_item()
                            .kind
                        else {
                            return;
                        };
                        let Some(trait_def_id) = trait_ref.trait_def_id() else {
                            return;
                        };
                        let Some((bound_vars, assoc_item)) = BoundVarContext::supertrait_hrtb_vars(
                            self.tcx,
                            trait_def_id,
                            item_segment.ident,
                            ty::AssocKind::Fn,
                        ) else {
                            return;
                        };
                        (bound_vars, assoc_item.def_id, item_segment)
                    }
                    _ => return,
                }
            }

            _ => return,
        };

        // Append the early-bound vars on the function, and then the late-bound ones.
        // We actually turn type parameters into higher-ranked types here, but we
        // deny them later in HIR lowering.
        bound_vars.extend(self.tcx.generics_of(item_def_id).own_params.iter().map(|param| {
            match param.kind {
                ty::GenericParamDefKind::Lifetime => ty::BoundVariableKind::Region(
                    ty::BoundRegionKind::BrNamed(param.def_id, param.name),
                ),
                ty::GenericParamDefKind::Type { .. } => {
                    ty::BoundVariableKind::Ty(ty::BoundTyKind::Param(param.def_id, param.name))
                }
                ty::GenericParamDefKind::Const { .. } => ty::BoundVariableKind::Const,
            }
        }));
        bound_vars.extend(self.tcx.fn_sig(item_def_id).instantiate_identity().bound_vars());

        // SUBTLE: Stash the old bound vars onto the *item segment* before appending
        // the new bound vars. We do this because we need to know how many bound vars
        // are present on the binder explicitly (i.e. not return-type-notation vars)
        // to do bound var shifting correctly in HIR lowering.
        //
        // For example, in `where for<'a> <T as Trait<'a>>::method(..): Other`,
        // the `late_bound_vars` of the where clause predicate (i.e. this HIR ty's
        // parent) will include `'a` AND all the early- and late-bound vars of the
        // method. But when lowering the RTN type, we just want the list of vars
        // we used to resolve the trait ref. We explicitly stored those back onto
        // the item segment, since there's no other good place to put them.
        //
        // See where these vars are used in `HirTyLowerer::lower_ty_maybe_return_type_notation`.
        // And this is exercised in:
        // `tests/ui/associated-type-bounds/return-type-notation/higher-ranked-bound-works.rs`.
        let existing_bound_vars = self.map.late_bound_vars.get_mut(&hir_id.local_id).unwrap();
        let existing_bound_vars_saved = existing_bound_vars.clone();
        existing_bound_vars.extend(bound_vars);
        self.record_late_bound_vars(item_segment.hir_id, existing_bound_vars_saved);
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
    /// ```rust,ignore (pseudo-Rust)
    /// type Alias<'a, T> = <T as Trait<'a>>::Assoc;
    /// fn foo<'a>(_: Alias<'a, ()>) -> Alias<'a, ()> { ... }
    /// ```
    ///
    /// If we conservatively considered `'a` unconstrained then we could break users who had written code before
    /// we started correctly handling aliases. If we considered `'a` constrained then it would become late bound
    /// causing an error during HIR ty lowering as the `'a` is not constrained by the input type `<() as Trait<'a>>::Assoc`
    /// but appears in the output type `<() as Trait<'a>>::Assoc`.
    ///
    /// We must therefore "look into" the `Alias` to see whether we should consider `'a` constrained or not.
    ///
    /// See #100508 #85533 #47511 for additional context
    struct ConstrainedCollectorPostHirTyLowering {
        arg_is_constrained: Box<[bool]>,
    }

    use ty::Ty;
    impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for ConstrainedCollectorPostHirTyLowering {
        fn visit_ty(&mut self, t: Ty<'tcx>) {
            match t.kind() {
                ty::Param(param_ty) => {
                    self.arg_is_constrained[param_ty.index as usize] = true;
                }
                ty::Alias(ty::Projection | ty::Inherent, _) => return,
                _ => (),
            }
            t.super_visit_with(self)
        }

        fn visit_const(&mut self, _: ty::Const<'tcx>) {}

        fn visit_region(&mut self, r: ty::Region<'tcx>) {
            debug!("r={:?}", r.kind());
            if let ty::RegionKind::ReEarlyParam(region) = r.kind() {
                self.arg_is_constrained[region.index as usize] = true;
            }
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
                    // See comments on `ConstrainedCollectorPostHirTyLowering` for why this arm does not
                    // just consider args to be unconstrained.
                    let generics = self.tcx.generics_of(alias_def);
                    let mut walker = ConstrainedCollectorPostHirTyLowering {
                        arg_is_constrained: vec![false; generics.own_params.len()]
                            .into_boxed_slice(),
                    };
                    walker.visit_ty(self.tcx.type_of(alias_def).instantiate_identity());

                    match segments.last() {
                        Some(hir::PathSegment { args: Some(args), .. }) => {
                            let tcx = self.tcx;
                            for constrained_arg in
                                args.args.iter().enumerate().flat_map(|(n, arg)| {
                                    match walker.arg_is_constrained.get(n) {
                                        Some(true) => Some(arg),
                                        Some(false) => None,
                                        None => {
                                            tcx.dcx().span_delayed_bug(
                                                *span,
                                                format!(
                                                    "Incorrect generic arg count for alias {alias_def:?}"
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

fn deny_non_region_late_bound(
    tcx: TyCtxt<'_>,
    bound_vars: &mut FxIndexMap<LocalDefId, ResolvedArg>,
    where_: &str,
) {
    let mut first = true;

    for (var, arg) in bound_vars {
        let Node::GenericParam(param) = tcx.hir_node_by_def_id(*var) else {
            span_bug!(tcx.def_span(*var), "expected bound-var def-id to resolve to param");
        };

        let what = match param.kind {
            hir::GenericParamKind::Type { .. } => "type",
            hir::GenericParamKind::Const { .. } => "const",
            hir::GenericParamKind::Lifetime { .. } => continue,
        };

        let diag = tcx.dcx().struct_span_err(
            param.span,
            format!("late-bound {what} parameter not allowed on {where_}"),
        );

        let guar = diag.emit_unless(!tcx.features().non_lifetime_binders || !first);

        first = false;
        *arg = ResolvedArg::Error(guar);
    }
}
