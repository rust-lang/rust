use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{self as hir, Expr, ImplItem, Item, Node, TraitItem, def, intravisit};
use rustc_middle::bug;
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::{self, DefiningScopeKind, Ty, TyCtxt, TypeVisitableExt};
use rustc_trait_selection::opaque_types::report_item_does_not_constrain_error;
use tracing::{debug, instrument, trace};

use crate::errors::UnconstrainedOpaqueType;

/// Checks "defining uses" of opaque `impl Trait` in associated types.
/// These can only be defined by associated items of the same trait.
#[instrument(skip(tcx), level = "debug")]
pub(super) fn find_opaque_ty_constraints_for_impl_trait_in_assoc_type(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
    opaque_types_from: DefiningScopeKind,
) -> Ty<'_> {
    let mut parent_def_id = def_id;
    while tcx.def_kind(parent_def_id) == def::DefKind::OpaqueTy {
        // Account for `type Alias = impl Trait<Foo = impl Trait>;` (#116031)
        parent_def_id = tcx.local_parent(parent_def_id);
    }
    let impl_def_id = tcx.local_parent(parent_def_id);
    match tcx.def_kind(impl_def_id) {
        DefKind::Impl { .. } => {}
        other => bug!("invalid impl trait in assoc type parent: {other:?}"),
    }

    let mut locator = TaitConstraintLocator { def_id, tcx, found: None, opaque_types_from };

    for &assoc_id in tcx.associated_item_def_ids(impl_def_id) {
        let assoc = tcx.associated_item(assoc_id);
        match assoc.kind {
            ty::AssocKind::Const { .. } | ty::AssocKind::Fn { .. } => {
                locator.check(assoc_id.expect_local())
            }
            // Associated types don't have bodies, so they can't constrain hidden types
            ty::AssocKind::Type { .. } => {}
        }
    }

    if let Some(hidden) = locator.found {
        hidden.ty
    } else {
        let guar = tcx.dcx().emit_err(UnconstrainedOpaqueType {
            span: tcx.def_span(def_id),
            name: tcx.item_ident(parent_def_id.to_def_id()),
            what: "impl",
        });
        Ty::new_error(tcx, guar)
    }
}

/// Checks "defining uses" of opaque `impl Trait` types to ensure that they meet the restrictions
/// laid for "higher-order pattern unification".
/// This ensures that inference is tractable.
/// In particular, definitions of opaque types can only use other generics as arguments,
/// and they cannot repeat an argument. Example:
///
/// ```ignore (illustrative)
/// type Foo<A, B> = impl Bar<A, B>;
///
/// // Okay -- `Foo` is applied to two distinct, generic types.
/// fn a<T, U>() -> Foo<T, U> { .. }
///
/// // Not okay -- `Foo` is applied to `T` twice.
/// fn b<T>() -> Foo<T, T> { .. }
///
/// // Not okay -- `Foo` is applied to a non-generic type.
/// fn b<T>() -> Foo<T, u32> { .. }
/// ```
#[instrument(skip(tcx), level = "debug")]
pub(super) fn find_opaque_ty_constraints_for_tait(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
    opaque_types_from: DefiningScopeKind,
) -> Ty<'_> {
    let mut locator = TaitConstraintLocator { def_id, tcx, found: None, opaque_types_from };

    tcx.hir_walk_toplevel_module(&mut locator);

    if let Some(hidden) = locator.found {
        hidden.ty
    } else {
        let mut parent_def_id = def_id;
        while tcx.def_kind(parent_def_id) == def::DefKind::OpaqueTy {
            // Account for `type Alias = impl Trait<Foo = impl Trait>;` (#116031)
            parent_def_id = tcx.local_parent(parent_def_id);
        }
        let guar = tcx.dcx().emit_err(UnconstrainedOpaqueType {
            span: tcx.def_span(def_id),
            name: tcx.item_ident(parent_def_id.to_def_id()),
            what: "crate",
        });
        Ty::new_error(tcx, guar)
    }
}

struct TaitConstraintLocator<'tcx> {
    tcx: TyCtxt<'tcx>,

    /// def_id of the opaque type whose defining uses are being checked
    def_id: LocalDefId,

    /// as we walk the defining uses, we are checking that all of them
    /// define the same hidden type. This variable is set to `Some`
    /// with the first type that we find, and then later types are
    /// checked against it (we also carry the span of that first
    /// type).
    found: Option<ty::OpaqueHiddenType<'tcx>>,

    opaque_types_from: DefiningScopeKind,
}

impl<'tcx> TaitConstraintLocator<'tcx> {
    fn insert_found(&mut self, hidden_ty: ty::OpaqueHiddenType<'tcx>) {
        if let Some(prev) = &mut self.found {
            if hidden_ty.ty != prev.ty {
                let (Ok(guar) | Err(guar)) =
                    prev.build_mismatch_error(&hidden_ty, self.tcx).map(|d| d.emit());
                prev.ty = Ty::new_error(self.tcx, guar);
            }
        } else {
            self.found = Some(hidden_ty);
        }
    }

    fn non_defining_use_in_defining_scope(&mut self, item_def_id: LocalDefId) {
        // We make sure that all opaque types get defined while
        // type checking the defining scope, so this error is unreachable
        // with the new solver.
        assert!(!self.tcx.next_trait_solver_globally());
        let guar = report_item_does_not_constrain_error(self.tcx, item_def_id, self.def_id, None);
        self.insert_found(ty::OpaqueHiddenType::new_error(self.tcx, guar));
    }

    #[instrument(skip(self), level = "debug")]
    fn check(&mut self, item_def_id: LocalDefId) {
        // Don't try to check items that cannot possibly constrain the type.
        let tcx = self.tcx;
        if !tcx.has_typeck_results(item_def_id) {
            debug!("no constraint: no typeck results");
            return;
        }

        let opaque_types_defined_by = tcx.opaque_types_defined_by(item_def_id);
        // Don't try to check items that cannot possibly constrain the type.
        if !opaque_types_defined_by.contains(&self.def_id) {
            debug!("no constraint: no opaque types defined");
            return;
        }

        // Function items with `_` in their return type already emit an error, skip any
        // "non-defining use" errors for them.
        // Note that we use `Node::fn_sig` instead of `Node::fn_decl` here, because the former
        // excludes closures, which are allowed to have `_` in their return type.
        let hir_node = tcx.hir_node_by_def_id(item_def_id);
        debug_assert!(
            !matches!(hir_node, Node::ForeignItem(..)),
            "foreign items cannot constrain opaque types",
        );
        if let Some(hir_sig) = hir_node.fn_sig()
            && hir_sig.decl.output.is_suggestable_infer_ty().is_some()
        {
            let guar = self.tcx.dcx().span_delayed_bug(
                hir_sig.decl.output.span(),
                "inferring return types and opaque types do not mix well",
            );
            self.found = Some(ty::OpaqueHiddenType::new_error(tcx, guar));
            return;
        }

        match self.opaque_types_from {
            DefiningScopeKind::HirTypeck => {
                let tables = tcx.typeck(item_def_id);
                if let Some(guar) = tables.tainted_by_errors {
                    self.insert_found(ty::OpaqueHiddenType::new_error(tcx, guar));
                } else if let Some(&hidden_type) = tables.hidden_types.get(&self.def_id) {
                    self.insert_found(hidden_type);
                } else {
                    self.non_defining_use_in_defining_scope(item_def_id);
                }
            }
            DefiningScopeKind::MirBorrowck => match tcx.mir_borrowck(item_def_id) {
                Err(guar) => self.insert_found(ty::OpaqueHiddenType::new_error(tcx, guar)),
                Ok(hidden_types) => {
                    if let Some(&hidden_type) = hidden_types.0.get(&self.def_id) {
                        debug!(?hidden_type, "found constraint");
                        self.insert_found(hidden_type);
                    } else if let Err(guar) = tcx
                        .type_of_opaque_hir_typeck(self.def_id)
                        .instantiate_identity()
                        .error_reported()
                    {
                        self.insert_found(ty::OpaqueHiddenType::new_error(tcx, guar));
                    } else {
                        self.non_defining_use_in_defining_scope(item_def_id);
                    }
                }
            },
        }
    }
}

impl<'tcx> intravisit::Visitor<'tcx> for TaitConstraintLocator<'tcx> {
    type NestedFilter = nested_filter::All;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }
    fn visit_expr(&mut self, ex: &'tcx Expr<'tcx>) {
        intravisit::walk_expr(self, ex);
    }
    fn visit_item(&mut self, it: &'tcx Item<'tcx>) {
        trace!(?it.owner_id);
        self.check(it.owner_id.def_id);
        intravisit::walk_item(self, it);
    }
    fn visit_impl_item(&mut self, it: &'tcx ImplItem<'tcx>) {
        trace!(?it.owner_id);
        self.check(it.owner_id.def_id);
        intravisit::walk_impl_item(self, it);
    }
    fn visit_trait_item(&mut self, it: &'tcx TraitItem<'tcx>) {
        trace!(?it.owner_id);
        self.check(it.owner_id.def_id);
        intravisit::walk_trait_item(self, it);
    }
    fn visit_foreign_item(&mut self, it: &'tcx hir::ForeignItem<'tcx>) {
        trace!(?it.owner_id);
        assert_ne!(it.owner_id.def_id, self.def_id);
        // No need to call `check`, as we do not run borrowck on foreign items.
        intravisit::walk_foreign_item(self, it);
    }
}

pub(super) fn find_opaque_ty_constraints_for_rpit<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    owner_def_id: LocalDefId,
    opaque_types_from: DefiningScopeKind,
) -> Ty<'tcx> {
    match opaque_types_from {
        DefiningScopeKind::HirTypeck => {
            let tables = tcx.typeck(owner_def_id);
            if let Some(guar) = tables.tainted_by_errors {
                Ty::new_error(tcx, guar)
            } else if let Some(hidden_ty) = tables.hidden_types.get(&def_id) {
                hidden_ty.ty
            } else {
                assert!(!tcx.next_trait_solver_globally());
                // We failed to resolve the opaque type or it
                // resolves to itself. We interpret this as the
                // no values of the hidden type ever being constructed,
                // so we can just make the hidden type be `!`.
                // For backwards compatibility reasons, we fall back to
                // `()` until we the diverging default is changed.
                Ty::new_diverging_default(tcx)
            }
        }
        DefiningScopeKind::MirBorrowck => match tcx.mir_borrowck(owner_def_id) {
            Ok(hidden_types) => {
                if let Some(hidden_ty) = hidden_types.0.get(&def_id) {
                    hidden_ty.ty
                } else {
                    let hir_ty = tcx.type_of_opaque_hir_typeck(def_id).instantiate_identity();
                    if let Err(guar) = hir_ty.error_reported() {
                        Ty::new_error(tcx, guar)
                    } else {
                        assert!(!tcx.next_trait_solver_globally());
                        hir_ty
                    }
                }
            }
            Err(guar) => Ty::new_error(tcx, guar),
        },
    }
}
