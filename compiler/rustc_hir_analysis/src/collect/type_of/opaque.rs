use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{self as hir, Expr, ImplItem, Item, Node, TraitItem, def};
use rustc_middle::bug;
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitableExt};
use rustc_span::DUMMY_SP;
use tracing::{debug, instrument, trace};

use crate::errors::{TaitForwardCompat, TaitForwardCompat2, UnconstrainedOpaqueType};

/// Checks "defining uses" of opaque `impl Trait` in associated types.
/// These can only be defined by associated items of the same trait.
#[instrument(skip(tcx), level = "debug")]
pub(super) fn find_opaque_ty_constraints_for_impl_trait_in_assoc_type(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
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

    let mut locator = TaitConstraintLocator { def_id, tcx, found: None, typeck_types: vec![] };

    for &assoc_id in tcx.associated_item_def_ids(impl_def_id) {
        let assoc = tcx.associated_item(assoc_id);
        match assoc.kind {
            ty::AssocKind::Const | ty::AssocKind::Fn => locator.check(assoc_id.expect_local()),
            // Associated types don't have bodies, so they can't constrain hidden types
            ty::AssocKind::Type => {}
        }
    }

    if let Some(hidden) = locator.found {
        // Only check against typeck if we didn't already error
        if !hidden.ty.references_error() {
            for concrete_type in locator.typeck_types {
                if concrete_type.ty != tcx.erase_regions(hidden.ty) {
                    if let Ok(d) = hidden.build_mismatch_error(&concrete_type, tcx) {
                        d.emit();
                    }
                }
            }
        }

        hidden.ty
    } else {
        let reported = tcx.dcx().emit_err(UnconstrainedOpaqueType {
            span: tcx.def_span(def_id),
            name: tcx.item_ident(parent_def_id.to_def_id()),
            what: "impl",
        });
        Ty::new_error(tcx, reported)
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
pub(super) fn find_opaque_ty_constraints_for_tait(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Ty<'_> {
    let hir_id = tcx.local_def_id_to_hir_id(def_id);
    let scope = tcx.hir().get_defining_scope(hir_id);
    let mut locator = TaitConstraintLocator { def_id, tcx, found: None, typeck_types: vec![] };

    debug!(?scope);

    if scope == hir::CRATE_HIR_ID {
        tcx.hir().walk_toplevel_module(&mut locator);
    } else {
        trace!("scope={:#?}", tcx.hir_node(scope));
        match tcx.hir_node(scope) {
            // We explicitly call `visit_*` methods, instead of using `intravisit::walk_*` methods
            // This allows our visitor to process the defining item itself, causing
            // it to pick up any 'sibling' defining uses.
            //
            // For example, this code:
            // ```
            // fn foo() {
            //     type Blah = impl Debug;
            //     let my_closure = || -> Blah { true };
            // }
            // ```
            //
            // requires us to explicitly process `foo()` in order
            // to notice the defining usage of `Blah`.
            Node::Item(it) => locator.visit_item(it),
            Node::ImplItem(it) => locator.visit_impl_item(it),
            Node::TraitItem(it) => locator.visit_trait_item(it),
            Node::ForeignItem(it) => locator.visit_foreign_item(it),
            other => bug!("{:?} is not a valid scope for an opaque type item", other),
        }
    }

    if let Some(hidden) = locator.found {
        // Only check against typeck if we didn't already error
        if !hidden.ty.references_error() {
            for concrete_type in locator.typeck_types {
                if concrete_type.ty != tcx.erase_regions(hidden.ty) {
                    if let Ok(d) = hidden.build_mismatch_error(&concrete_type, tcx) {
                        d.emit();
                    }
                }
            }
        }

        hidden.ty
    } else {
        let mut parent_def_id = def_id;
        while tcx.def_kind(parent_def_id) == def::DefKind::OpaqueTy {
            // Account for `type Alias = impl Trait<Foo = impl Trait>;` (#116031)
            parent_def_id = tcx.local_parent(parent_def_id);
        }
        let reported = tcx.dcx().emit_err(UnconstrainedOpaqueType {
            span: tcx.def_span(def_id),
            name: tcx.item_ident(parent_def_id.to_def_id()),
            what: match tcx.hir_node(scope) {
                _ if scope == hir::CRATE_HIR_ID => "module",
                Node::Item(hir::Item { kind: hir::ItemKind::Mod(_), .. }) => "module",
                Node::Item(hir::Item { kind: hir::ItemKind::Impl(_), .. }) => "impl",
                _ => "item",
            },
        });
        Ty::new_error(tcx, reported)
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

    /// In the presence of dead code, typeck may figure out a hidden type
    /// while borrowck will not. We collect these cases here and check at
    /// the end that we actually found a type that matches (modulo regions).
    typeck_types: Vec<ty::OpaqueHiddenType<'tcx>>,
}

impl TaitConstraintLocator<'_> {
    #[instrument(skip(self), level = "debug")]
    fn check(&mut self, item_def_id: LocalDefId) {
        // Don't try to check items that cannot possibly constrain the type.
        if !self.tcx.has_typeck_results(item_def_id) {
            debug!("no constraint: no typeck results");
            return;
        }

        // Function items with `_` in their return type already emit an error, skip any
        // "non-defining use" errors for them.
        // Note that we use `Node::fn_sig` instead of `Node::fn_decl` here, because the former
        // excludes closures, which are allowed to have `_` in their return type.
        let hir_node = self.tcx.hir_node_by_def_id(item_def_id);
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
            self.found =
                Some(ty::OpaqueHiddenType { span: DUMMY_SP, ty: Ty::new_error(self.tcx, guar) });
            return;
        }

        // Calling `mir_borrowck` can lead to cycle errors through
        // const-checking, avoid calling it if we don't have to.
        // ```rust
        // type Foo = impl Fn() -> usize; // when computing type for this
        // const fn bar() -> Foo {
        //     || 0usize
        // }
        // const BAZR: Foo = bar(); // we would mir-borrowck this, causing cycles
        // // because we again need to reveal `Foo` so we can check whether the
        // // constant does not contain interior mutability.
        // ```
        let tables = self.tcx.typeck(item_def_id);
        if let Some(guar) = tables.tainted_by_errors {
            self.found =
                Some(ty::OpaqueHiddenType { span: DUMMY_SP, ty: Ty::new_error(self.tcx, guar) });
            return;
        }

        let opaque_types_defined_by = self.tcx.opaque_types_defined_by(item_def_id);

        let mut constrained = false;
        for (&opaque_type_key, &hidden_type) in &tables.concrete_opaque_types {
            if opaque_type_key.def_id != self.def_id {
                continue;
            }
            constrained = true;

            if !opaque_types_defined_by.contains(&self.def_id) {
                let guar = self.tcx.dcx().emit_err(TaitForwardCompat {
                    span: hidden_type.span,
                    item_span: self
                        .tcx
                        .def_ident_span(item_def_id)
                        .unwrap_or_else(|| self.tcx.def_span(item_def_id)),
                });
                // Avoid "opaque type not constrained" errors on the opaque itself.
                self.found = Some(ty::OpaqueHiddenType {
                    span: DUMMY_SP,
                    ty: Ty::new_error(self.tcx, guar),
                });
            }
            let concrete_type =
                self.tcx.erase_regions(hidden_type.remap_generic_params_to_declaration_params(
                    opaque_type_key,
                    self.tcx,
                    true,
                ));
            if self.typeck_types.iter().all(|prev| prev.ty != concrete_type.ty) {
                self.typeck_types.push(concrete_type);
            }
        }

        if !constrained {
            debug!("no constraints in typeck results");
            if opaque_types_defined_by.contains(&self.def_id) {
                let guar = self.tcx.dcx().emit_err(TaitForwardCompat2 {
                    span: self
                        .tcx
                        .def_ident_span(item_def_id)
                        .unwrap_or_else(|| self.tcx.def_span(item_def_id)),
                    opaque_type_span: self.tcx.def_span(self.def_id),
                    opaque_type: self.tcx.def_path_str(self.def_id),
                });
                // Avoid "opaque type not constrained" errors on the opaque itself.
                self.found = Some(ty::OpaqueHiddenType {
                    span: DUMMY_SP,
                    ty: Ty::new_error(self.tcx, guar),
                });
            }
            return;
        };

        // Use borrowck to get the type with unerased regions.
        let borrowck_results = &self.tcx.mir_borrowck(item_def_id);

        // If the body was tainted, then assume the opaque may have been constrained and just set it to error.
        if let Some(guar) = borrowck_results.tainted_by_errors {
            self.found =
                Some(ty::OpaqueHiddenType { span: DUMMY_SP, ty: Ty::new_error(self.tcx, guar) });
            return;
        }

        debug!(?borrowck_results.concrete_opaque_types);
        if let Some(&concrete_type) = borrowck_results.concrete_opaque_types.get(&self.def_id) {
            debug!(?concrete_type, "found constraint");
            if let Some(prev) = &mut self.found {
                if concrete_type.ty != prev.ty {
                    let (Ok(guar) | Err(guar)) =
                        prev.build_mismatch_error(&concrete_type, self.tcx).map(|d| d.emit());
                    prev.ty = Ty::new_error(self.tcx, guar);
                }
            } else {
                self.found = Some(concrete_type);
            }
        }
    }
}

impl<'tcx> intravisit::Visitor<'tcx> for TaitConstraintLocator<'tcx> {
    type NestedFilter = nested_filter::All;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }
    fn visit_expr(&mut self, ex: &'tcx Expr<'tcx>) {
        if let hir::ExprKind::Closure(closure) = ex.kind {
            self.check(closure.def_id);
        }
        intravisit::walk_expr(self, ex);
    }
    fn visit_item(&mut self, it: &'tcx Item<'tcx>) {
        trace!(?it.owner_id);
        // The opaque type itself or its children are not within its reveal scope.
        if it.owner_id.def_id != self.def_id {
            self.check(it.owner_id.def_id);
            intravisit::walk_item(self, it);
        }
    }
    fn visit_impl_item(&mut self, it: &'tcx ImplItem<'tcx>) {
        trace!(?it.owner_id);
        // The opaque type itself or its children are not within its reveal scope.
        if it.owner_id.def_id != self.def_id {
            self.check(it.owner_id.def_id);
            intravisit::walk_impl_item(self, it);
        }
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
) -> Ty<'tcx> {
    let tables = tcx.typeck(owner_def_id);

    // Check that all of the opaques we inferred during HIR are compatible.
    // FIXME: We explicitly don't check that the types inferred during HIR
    // typeck are compatible with the one that we infer during borrowck,
    // because that one actually sometimes has consts evaluated eagerly so
    // using strict type equality will fail.
    let mut hir_opaque_ty: Option<ty::OpaqueHiddenType<'tcx>> = None;
    if tables.tainted_by_errors.is_none() {
        for (&opaque_type_key, &hidden_type) in &tables.concrete_opaque_types {
            if opaque_type_key.def_id != def_id {
                continue;
            }
            let concrete_type = tcx.erase_regions(
                hidden_type.remap_generic_params_to_declaration_params(opaque_type_key, tcx, true),
            );
            if let Some(prev) = &mut hir_opaque_ty {
                if concrete_type.ty != prev.ty {
                    if let Ok(d) = prev.build_mismatch_error(&concrete_type, tcx) {
                        d.emit();
                    }
                }
            } else {
                hir_opaque_ty = Some(concrete_type);
            }
        }
    }

    let mir_opaque_ty = tcx.mir_borrowck(owner_def_id).concrete_opaque_types.get(&def_id).copied();
    if let Some(mir_opaque_ty) = mir_opaque_ty {
        if mir_opaque_ty.references_error() {
            return mir_opaque_ty.ty;
        }

        debug!(?owner_def_id);
        let mut locator = RpitConstraintChecker { def_id, tcx, found: mir_opaque_ty };

        match tcx.hir_node_by_def_id(owner_def_id) {
            Node::Item(it) => intravisit::walk_item(&mut locator, it),
            Node::ImplItem(it) => intravisit::walk_impl_item(&mut locator, it),
            Node::TraitItem(it) => intravisit::walk_trait_item(&mut locator, it),
            other => bug!("{:?} is not a valid scope for an opaque type item", other),
        }

        mir_opaque_ty.ty
    } else if let Some(guar) = tables.tainted_by_errors {
        // Some error in the owner fn prevented us from populating
        // the `concrete_opaque_types` table.
        Ty::new_error(tcx, guar)
    } else {
        // Fall back to the RPIT we inferred during HIR typeck
        if let Some(hir_opaque_ty) = hir_opaque_ty {
            hir_opaque_ty.ty
        } else {
            // We failed to resolve the opaque type or it
            // resolves to itself. We interpret this as the
            // no values of the hidden type ever being constructed,
            // so we can just make the hidden type be `!`.
            // For backwards compatibility reasons, we fall back to
            // `()` until we the diverging default is changed.
            Ty::new_diverging_default(tcx)
        }
    }
}

struct RpitConstraintChecker<'tcx> {
    tcx: TyCtxt<'tcx>,

    /// def_id of the opaque type whose defining uses are being checked
    def_id: LocalDefId,

    found: ty::OpaqueHiddenType<'tcx>,
}

impl RpitConstraintChecker<'_> {
    #[instrument(skip(self), level = "debug")]
    fn check(&self, def_id: LocalDefId) {
        // Use borrowck to get the type with unerased regions.
        let concrete_opaque_types = &self.tcx.mir_borrowck(def_id).concrete_opaque_types;
        debug!(?concrete_opaque_types);
        for (&def_id, &concrete_type) in concrete_opaque_types {
            if def_id != self.def_id {
                // Ignore constraints for other opaque types.
                continue;
            }

            debug!(?concrete_type, "found constraint");

            if concrete_type.ty != self.found.ty {
                if let Ok(d) = self.found.build_mismatch_error(&concrete_type, self.tcx) {
                    d.emit();
                }
            }
        }
    }
}

impl<'tcx> intravisit::Visitor<'tcx> for RpitConstraintChecker<'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }
    fn visit_expr(&mut self, ex: &'tcx Expr<'tcx>) {
        if let hir::ExprKind::Closure(closure) = ex.kind {
            self.check(closure.def_id);
        }
        intravisit::walk_expr(self, ex);
    }
    fn visit_item(&mut self, it: &'tcx Item<'tcx>) {
        trace!(?it.owner_id);
        // The opaque type itself or its children are not within its reveal scope.
        if it.owner_id.def_id != self.def_id {
            self.check(it.owner_id.def_id);
            intravisit::walk_item(self, it);
        }
    }
    fn visit_impl_item(&mut self, it: &'tcx ImplItem<'tcx>) {
        trace!(?it.owner_id);
        // The opaque type itself or its children are not within its reveal scope.
        if it.owner_id.def_id != self.def_id {
            self.check(it.owner_id.def_id);
            intravisit::walk_impl_item(self, it);
        }
    }
    fn visit_trait_item(&mut self, it: &'tcx TraitItem<'tcx>) {
        trace!(?it.owner_id);
        self.check(it.owner_id.def_id);
        intravisit::walk_trait_item(self, it);
    }
}
