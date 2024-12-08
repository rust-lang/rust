// Type resolution: the phase that finds all the types in the AST with
// unresolved type variables and replaces "ty_var" types with their
// generic parameters.

use std::mem;

use rustc_data_structures::unord::ExtendUnord;
use rustc_errors::{ErrorGuaranteed, StashKey};
use rustc_hir as hir;
use rustc_hir::HirId;
use rustc_hir::intravisit::{self, Visitor};
use rustc_middle::span_bug;
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::adjustment::{Adjust, Adjustment, PointerCoercion};
use rustc_middle::ty::fold::{TypeFoldable, TypeFolder, fold_regions};
use rustc_middle::ty::visit::TypeVisitableExt;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeSuperFoldable};
use rustc_span::Span;
use rustc_span::symbol::sym;
use rustc_trait_selection::error_reporting::infer::need_type_info::TypeAnnotationNeeded;
use rustc_trait_selection::solve;
use tracing::{debug, instrument};

use crate::FnCtxt;

///////////////////////////////////////////////////////////////////////////
// Entry point

// During type inference, partially inferred types are
// represented using Type variables (ty::Infer). These don't appear in
// the final TypeckResults since all of the types should have been
// inferred once typeck is done.
// When type inference is running however, having to update the typeck
// typeck results every time a new type is inferred would be unreasonably slow,
// so instead all of the replacement happens at the end in
// resolve_type_vars_in_body, which creates a new TypeTables which
// doesn't contain any inference types.
impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub(crate) fn resolve_type_vars_in_body(
        &self,
        body: &'tcx hir::Body<'tcx>,
    ) -> &'tcx ty::TypeckResults<'tcx> {
        let item_def_id = self.tcx.hir().body_owner_def_id(body.id());

        // This attribute causes us to dump some writeback information
        // in the form of errors, which is used for unit tests.
        let rustc_dump_user_args = self.tcx.has_attr(item_def_id, sym::rustc_dump_user_args);

        let mut wbcx = WritebackCx::new(self, body, rustc_dump_user_args);
        for param in body.params {
            wbcx.visit_node_id(param.pat.span, param.hir_id);
        }
        // Type only exists for constants and statics, not functions.
        match self.tcx.hir().body_owner_kind(item_def_id) {
            hir::BodyOwnerKind::Const { .. } | hir::BodyOwnerKind::Static(_) => {
                let item_hir_id = self.tcx.local_def_id_to_hir_id(item_def_id);
                wbcx.visit_node_id(body.value.span, item_hir_id);
            }
            hir::BodyOwnerKind::Closure | hir::BodyOwnerKind::Fn => (),
        }
        wbcx.visit_body(body);
        wbcx.visit_min_capture_map();
        wbcx.eval_closure_size();
        wbcx.visit_fake_reads_map();
        wbcx.visit_closures();
        wbcx.visit_liberated_fn_sigs();
        wbcx.visit_fru_field_types();
        wbcx.visit_opaque_types();
        wbcx.visit_coercion_casts();
        wbcx.visit_user_provided_tys();
        wbcx.visit_user_provided_sigs();
        wbcx.visit_coroutine_interior();
        wbcx.visit_offset_of_container_types();

        wbcx.typeck_results.rvalue_scopes =
            mem::take(&mut self.typeck_results.borrow_mut().rvalue_scopes);

        let used_trait_imports =
            mem::take(&mut self.typeck_results.borrow_mut().used_trait_imports);
        debug!("used_trait_imports({:?}) = {:?}", item_def_id, used_trait_imports);
        wbcx.typeck_results.used_trait_imports = used_trait_imports;

        wbcx.typeck_results.treat_byte_string_as_slice =
            mem::take(&mut self.typeck_results.borrow_mut().treat_byte_string_as_slice);

        debug!("writeback: typeck results for {:?} are {:#?}", item_def_id, wbcx.typeck_results);

        self.tcx.arena.alloc(wbcx.typeck_results)
    }
}

///////////////////////////////////////////////////////////////////////////
// The Writeback context. This visitor walks the HIR, checking the
// fn-specific typeck results to find references to types or regions. It
// resolves those regions to remove inference variables and writes the
// final result back into the master typeck results in the tcx. Here and
// there, it applies a few ad-hoc checks that were not convenient to
// do elsewhere.

struct WritebackCx<'cx, 'tcx> {
    fcx: &'cx FnCtxt<'cx, 'tcx>,

    typeck_results: ty::TypeckResults<'tcx>,

    body: &'tcx hir::Body<'tcx>,

    rustc_dump_user_args: bool,
}

impl<'cx, 'tcx> WritebackCx<'cx, 'tcx> {
    fn new(
        fcx: &'cx FnCtxt<'cx, 'tcx>,
        body: &'tcx hir::Body<'tcx>,
        rustc_dump_user_args: bool,
    ) -> WritebackCx<'cx, 'tcx> {
        let owner = body.id().hir_id.owner;

        let mut wbcx = WritebackCx {
            fcx,
            typeck_results: ty::TypeckResults::new(owner),
            body,
            rustc_dump_user_args,
        };

        // HACK: We specifically don't want the (opaque) error from tainting our
        // inference context. That'll prevent us from doing opaque type inference
        // later on in borrowck, which affects diagnostic spans pretty negatively.
        if let Some(e) = fcx.tainted_by_errors() {
            wbcx.typeck_results.tainted_by_errors = Some(e);
        }

        wbcx
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.fcx.tcx
    }

    fn write_ty_to_typeck_results(&mut self, hir_id: HirId, ty: Ty<'tcx>) {
        debug!("write_ty_to_typeck_results({:?}, {:?})", hir_id, ty);
        assert!(
            !ty.has_infer() && !ty.has_placeholders() && !ty.has_free_regions(),
            "{ty} can't be put into typeck results"
        );
        self.typeck_results.node_types_mut().insert(hir_id, ty);
    }

    // Hacky hack: During type-checking, we treat *all* operators
    // as potentially overloaded. But then, during writeback, if
    // we observe that something like `a+b` is (known to be)
    // operating on scalars, we clear the overload.
    fn fix_scalar_builtin_expr(&mut self, e: &hir::Expr<'_>) {
        match e.kind {
            hir::ExprKind::Unary(hir::UnOp::Neg | hir::UnOp::Not, inner) => {
                let inner_ty = self.typeck_results.node_type(inner.hir_id);

                if inner_ty.is_scalar() {
                    self.typeck_results.type_dependent_defs_mut().remove(e.hir_id);
                    self.typeck_results.node_args_mut().remove(e.hir_id);
                }
            }
            hir::ExprKind::Binary(ref op, lhs, rhs) | hir::ExprKind::AssignOp(ref op, lhs, rhs) => {
                let lhs_ty = self.typeck_results.node_type(lhs.hir_id);
                let rhs_ty = self.typeck_results.node_type(rhs.hir_id);

                if lhs_ty.is_scalar() && rhs_ty.is_scalar() {
                    self.typeck_results.type_dependent_defs_mut().remove(e.hir_id);
                    self.typeck_results.node_args_mut().remove(e.hir_id);

                    match e.kind {
                        hir::ExprKind::Binary(..) => {
                            if !op.node.is_by_value() {
                                let mut adjustments = self.typeck_results.adjustments_mut();
                                if let Some(a) = adjustments.get_mut(lhs.hir_id) {
                                    a.pop();
                                }
                                if let Some(a) = adjustments.get_mut(rhs.hir_id) {
                                    a.pop();
                                }
                            }
                        }
                        hir::ExprKind::AssignOp(..)
                            if let Some(a) =
                                self.typeck_results.adjustments_mut().get_mut(lhs.hir_id) =>
                        {
                            a.pop();
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    // (ouz-a 1005988): Normally `[T] : std::ops::Index<usize>` should be normalized
    // into [T] but currently `Where` clause stops the normalization process for it,
    // here we compare types of expr and base in a code without `Where` clause they would be equal
    // if they are not we don't modify the expr, hence we bypass the ICE
    fn is_builtin_index(
        &mut self,
        e: &hir::Expr<'_>,
        base_ty: Ty<'tcx>,
        index_ty: Ty<'tcx>,
    ) -> bool {
        if let Some(elem_ty) = base_ty.builtin_index()
            && let Some(exp_ty) = self.typeck_results.expr_ty_opt(e)
        {
            elem_ty == exp_ty && index_ty == self.fcx.tcx.types.usize
        } else {
            false
        }
    }

    // Similar to operators, indexing is always assumed to be overloaded
    // Here, correct cases where an indexing expression can be simplified
    // to use builtin indexing because the index type is known to be
    // usize-ish
    fn fix_index_builtin_expr(&mut self, e: &hir::Expr<'_>) {
        if let hir::ExprKind::Index(ref base, ref index, _) = e.kind {
            // All valid indexing looks like this; might encounter non-valid indexes at this point.
            let base_ty = self.typeck_results.expr_ty_adjusted(base);
            if let ty::Ref(_, base_ty_inner, _) = *base_ty.kind() {
                let index_ty = self.typeck_results.expr_ty_adjusted(index);
                if self.is_builtin_index(e, base_ty_inner, index_ty) {
                    // Remove the method call record
                    self.typeck_results.type_dependent_defs_mut().remove(e.hir_id);
                    self.typeck_results.node_args_mut().remove(e.hir_id);

                    if let Some(a) = self.typeck_results.adjustments_mut().get_mut(base.hir_id) {
                        // Discard the need for a mutable borrow

                        // Extra adjustment made when indexing causes a drop
                        // of size information - we need to get rid of it
                        // Since this is "after" the other adjustment to be
                        // discarded, we do an extra `pop()`
                        if let Some(Adjustment {
                            kind: Adjust::Pointer(PointerCoercion::Unsize),
                            ..
                        }) = a.pop()
                        {
                            // So the borrow discard actually happens here
                            a.pop();
                        }
                    }
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Impl of Visitor for Resolver
//
// This is the master code which walks the AST. It delegates most of
// the heavy lifting to the generic visit and resolve functions
// below. In general, a function is made into a `visitor` if it must
// traffic in node-ids or update typeck results in the type context etc.

impl<'cx, 'tcx> Visitor<'tcx> for WritebackCx<'cx, 'tcx> {
    fn visit_expr(&mut self, e: &'tcx hir::Expr<'tcx>) {
        match e.kind {
            hir::ExprKind::Closure(&hir::Closure { body, .. }) => {
                let body = self.fcx.tcx.hir().body(body);
                for param in body.params {
                    self.visit_node_id(e.span, param.hir_id);
                }

                self.visit_body(body);
            }
            hir::ExprKind::Struct(_, fields, _) => {
                for field in fields {
                    self.visit_field_id(field.hir_id);
                }
            }
            hir::ExprKind::Field(..) | hir::ExprKind::OffsetOf(..) => {
                self.visit_field_id(e.hir_id);
            }
            hir::ExprKind::ConstBlock(anon_const) => {
                self.visit_node_id(e.span, anon_const.hir_id);

                let body = self.tcx().hir().body(anon_const.body);
                self.visit_body(body);
            }
            _ => {}
        }

        self.visit_node_id(e.span, e.hir_id);
        intravisit::walk_expr(self, e);

        self.fix_scalar_builtin_expr(e);
        self.fix_index_builtin_expr(e);
    }

    fn visit_generic_param(&mut self, p: &'tcx hir::GenericParam<'tcx>) {
        match &p.kind {
            hir::GenericParamKind::Lifetime { .. } => {
                // Nothing to write back here
            }
            hir::GenericParamKind::Type { .. } | hir::GenericParamKind::Const { .. } => {
                self.tcx()
                    .dcx()
                    .span_delayed_bug(p.span, format!("unexpected generic param: {p:?}"));
            }
        }
    }

    fn visit_block(&mut self, b: &'tcx hir::Block<'tcx>) {
        self.visit_node_id(b.span, b.hir_id);
        intravisit::walk_block(self, b);
    }

    fn visit_pat(&mut self, p: &'tcx hir::Pat<'tcx>) {
        match p.kind {
            hir::PatKind::Binding(..) => {
                let typeck_results = self.fcx.typeck_results.borrow();
                if let Some(bm) =
                    typeck_results.extract_binding_mode(self.tcx().sess, p.hir_id, p.span)
                {
                    self.typeck_results.pat_binding_modes_mut().insert(p.hir_id, bm);
                }
            }
            hir::PatKind::Struct(_, fields, _) => {
                for field in fields {
                    self.visit_field_id(field.hir_id);
                }
            }
            _ => {}
        };

        self.visit_rust_2024_migration_desugared_pats(p.hir_id);
        self.visit_skipped_ref_pats(p.hir_id);
        self.visit_pat_adjustments(p.span, p.hir_id);

        self.visit_node_id(p.span, p.hir_id);
        intravisit::walk_pat(self, p);
    }

    fn visit_local(&mut self, l: &'tcx hir::LetStmt<'tcx>) {
        intravisit::walk_local(self, l);
        let var_ty = self.fcx.local_ty(l.span, l.hir_id);
        let var_ty = self.resolve(var_ty, &l.span);
        self.write_ty_to_typeck_results(l.hir_id, var_ty);
    }

    fn visit_ty(&mut self, hir_ty: &'tcx hir::Ty<'tcx>) {
        intravisit::walk_ty(self, hir_ty);
        // If there are type checking errors, Type privacy pass will stop,
        // so we may not get the type from hid_id, see #104513
        if let Some(ty) = self.fcx.node_ty_opt(hir_ty.hir_id) {
            let ty = self.resolve(ty, &hir_ty.span);
            self.write_ty_to_typeck_results(hir_ty.hir_id, ty);
        }
    }

    fn visit_infer(&mut self, inf: &'tcx hir::InferArg) {
        intravisit::walk_inf(self, inf);
        // Ignore cases where the inference is a const.
        if let Some(ty) = self.fcx.node_ty_opt(inf.hir_id) {
            let ty = self.resolve(ty, &inf.span);
            self.write_ty_to_typeck_results(inf.hir_id, ty);
        }
    }
}

impl<'cx, 'tcx> WritebackCx<'cx, 'tcx> {
    fn eval_closure_size(&mut self) {
        self.tcx().with_stable_hashing_context(|ref hcx| {
            let fcx_typeck_results = self.fcx.typeck_results.borrow();

            self.typeck_results.closure_size_eval = fcx_typeck_results
                .closure_size_eval
                .to_sorted(hcx, false)
                .into_iter()
                .map(|(&closure_def_id, data)| {
                    let closure_hir_id = self.tcx().local_def_id_to_hir_id(closure_def_id);
                    let data = self.resolve(*data, &closure_hir_id);
                    (closure_def_id, data)
                })
                .collect();
        })
    }

    fn visit_min_capture_map(&mut self) {
        self.tcx().with_stable_hashing_context(|ref hcx| {
            let fcx_typeck_results = self.fcx.typeck_results.borrow();

            self.typeck_results.closure_min_captures = fcx_typeck_results
                .closure_min_captures
                .to_sorted(hcx, false)
                .into_iter()
                .map(|(&closure_def_id, root_min_captures)| {
                    let root_var_map_wb = root_min_captures
                        .iter()
                        .map(|(var_hir_id, min_list)| {
                            let min_list_wb = min_list
                                .iter()
                                .map(|captured_place| {
                                    let locatable =
                                        captured_place.info.path_expr_id.unwrap_or_else(|| {
                                            self.tcx().local_def_id_to_hir_id(closure_def_id)
                                        });
                                    self.resolve(captured_place.clone(), &locatable)
                                })
                                .collect();
                            (*var_hir_id, min_list_wb)
                        })
                        .collect();
                    (closure_def_id, root_var_map_wb)
                })
                .collect();
        })
    }

    fn visit_fake_reads_map(&mut self) {
        self.tcx().with_stable_hashing_context(move |ref hcx| {
            let fcx_typeck_results = self.fcx.typeck_results.borrow();

            self.typeck_results.closure_fake_reads = fcx_typeck_results
                .closure_fake_reads
                .to_sorted(hcx, true)
                .into_iter()
                .map(|(&closure_def_id, fake_reads)| {
                    let resolved_fake_reads = fake_reads
                        .iter()
                        .map(|(place, cause, hir_id)| {
                            let locatable = self.tcx().local_def_id_to_hir_id(closure_def_id);
                            let resolved_fake_read = self.resolve(place.clone(), &locatable);
                            (resolved_fake_read, *cause, *hir_id)
                        })
                        .collect();

                    (closure_def_id, resolved_fake_reads)
                })
                .collect();
        });
    }

    fn visit_closures(&mut self) {
        let fcx_typeck_results = self.fcx.typeck_results.borrow();
        assert_eq!(fcx_typeck_results.hir_owner, self.typeck_results.hir_owner);
        let common_hir_owner = fcx_typeck_results.hir_owner;

        let fcx_closure_kind_origins =
            fcx_typeck_results.closure_kind_origins().items_in_stable_order();

        for (local_id, origin) in fcx_closure_kind_origins {
            let hir_id = HirId { owner: common_hir_owner, local_id };
            let place_span = origin.0;
            let place = self.resolve(origin.1.clone(), &place_span);
            self.typeck_results.closure_kind_origins_mut().insert(hir_id, (place_span, place));
        }
    }

    fn visit_coercion_casts(&mut self) {
        let fcx_typeck_results = self.fcx.typeck_results.borrow();

        assert_eq!(fcx_typeck_results.hir_owner, self.typeck_results.hir_owner);

        let fcx_coercion_casts = fcx_typeck_results.coercion_casts().to_sorted_stable_ord();
        for &local_id in fcx_coercion_casts {
            self.typeck_results.set_coercion_cast(local_id);
        }
    }

    fn visit_user_provided_tys(&mut self) {
        let fcx_typeck_results = self.fcx.typeck_results.borrow();
        assert_eq!(fcx_typeck_results.hir_owner, self.typeck_results.hir_owner);
        let common_hir_owner = fcx_typeck_results.hir_owner;

        if self.rustc_dump_user_args {
            let sorted_user_provided_types =
                fcx_typeck_results.user_provided_types().items_in_stable_order();

            let mut errors_buffer = Vec::new();
            for (local_id, c_ty) in sorted_user_provided_types {
                let hir_id = HirId { owner: common_hir_owner, local_id };

                if let ty::UserType::TypeOf(_, user_args) = c_ty.value {
                    // This is a unit-testing mechanism.
                    let span = self.tcx().hir().span(hir_id);
                    // We need to buffer the errors in order to guarantee a consistent
                    // order when emitting them.
                    let err =
                        self.tcx().dcx().struct_span_err(span, format!("user args: {user_args:?}"));
                    errors_buffer.push(err);
                }
            }

            if !errors_buffer.is_empty() {
                errors_buffer.sort_by_key(|diag| diag.span.primary_span());
                for err in errors_buffer {
                    err.emit();
                }
            }
        }

        self.typeck_results.user_provided_types_mut().extend(
            fcx_typeck_results.user_provided_types().items().map(|(local_id, c_ty)| {
                let hir_id = HirId { owner: common_hir_owner, local_id };

                if cfg!(debug_assertions) && c_ty.has_infer() {
                    span_bug!(
                        hir_id.to_span(self.fcx.tcx),
                        "writeback: `{:?}` has inference variables",
                        c_ty
                    );
                };

                (hir_id, *c_ty)
            }),
        );
    }

    fn visit_user_provided_sigs(&mut self) {
        let fcx_typeck_results = self.fcx.typeck_results.borrow();
        assert_eq!(fcx_typeck_results.hir_owner, self.typeck_results.hir_owner);

        self.typeck_results.user_provided_sigs.extend_unord(
            fcx_typeck_results.user_provided_sigs.items().map(|(&def_id, c_sig)| {
                if cfg!(debug_assertions) && c_sig.has_infer() {
                    span_bug!(
                        self.fcx.tcx.def_span(def_id),
                        "writeback: `{:?}` has inference variables",
                        c_sig
                    );
                };

                (def_id, *c_sig)
            }),
        );
    }

    fn visit_coroutine_interior(&mut self) {
        let fcx_typeck_results = self.fcx.typeck_results.borrow();
        assert_eq!(fcx_typeck_results.hir_owner, self.typeck_results.hir_owner);
        for (predicate, cause) in &fcx_typeck_results.coroutine_stalled_predicates {
            let (predicate, cause) = self.resolve((*predicate, cause.clone()), &cause.span);
            self.typeck_results.coroutine_stalled_predicates.insert((predicate, cause));
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_opaque_types(&mut self) {
        // We clone the opaques instead of stealing them here as they are still used for
        // normalization in the next generation trait solver.
        //
        // FIXME(-Znext-solver): Opaque types defined after this would simply get dropped
        // at the end of typeck. While this seems unlikely to happen in practice this
        // should still get fixed. Either by preventing writeback from defining new opaque
        // types or by using this function at the end of writeback and running it as a
        // fixpoint.
        let opaque_types = self.fcx.infcx.clone_opaque_types();
        for (opaque_type_key, decl) in opaque_types {
            let hidden_type = self.resolve(decl.hidden_type, &decl.hidden_type.span);
            let opaque_type_key = self.resolve(opaque_type_key, &decl.hidden_type.span);

            if let ty::Alias(ty::Opaque, alias_ty) = hidden_type.ty.kind()
                && alias_ty.def_id == opaque_type_key.def_id.to_def_id()
                && alias_ty.args == opaque_type_key.args
            {
                continue;
            }

            // Here we only detect impl trait definition conflicts when they
            // are equal modulo regions.
            if let Some(last_opaque_ty) =
                self.typeck_results.concrete_opaque_types.insert(opaque_type_key, hidden_type)
                && last_opaque_ty.ty != hidden_type.ty
            {
                assert!(!self.fcx.next_trait_solver());
                if let Ok(d) = hidden_type.build_mismatch_error(
                    &last_opaque_ty,
                    opaque_type_key.def_id,
                    self.tcx(),
                ) {
                    d.stash(
                        self.tcx().def_span(opaque_type_key.def_id),
                        StashKey::OpaqueHiddenTypeMismatch,
                    );
                }
            }
        }
    }

    fn visit_field_id(&mut self, hir_id: HirId) {
        if let Some(index) = self.fcx.typeck_results.borrow_mut().field_indices_mut().remove(hir_id)
        {
            self.typeck_results.field_indices_mut().insert(hir_id, index);
        }
    }

    #[instrument(skip(self, span), level = "debug")]
    fn visit_node_id(&mut self, span: Span, hir_id: HirId) {
        // Export associated path extensions and method resolutions.
        if let Some(def) =
            self.fcx.typeck_results.borrow_mut().type_dependent_defs_mut().remove(hir_id)
        {
            self.typeck_results.type_dependent_defs_mut().insert(hir_id, def);
        }

        // Resolve any borrowings for the node with id `node_id`
        self.visit_adjustments(span, hir_id);

        // Resolve the type of the node with id `node_id`
        let n_ty = self.fcx.node_ty(hir_id);
        let n_ty = self.resolve(n_ty, &span);
        self.write_ty_to_typeck_results(hir_id, n_ty);
        debug!(?n_ty);

        // Resolve any generic parameters
        if let Some(args) = self.fcx.typeck_results.borrow().node_args_opt(hir_id) {
            let args = self.resolve(args, &span);
            debug!("write_args_to_tcx({:?}, {:?})", hir_id, args);
            assert!(!args.has_infer() && !args.has_placeholders());
            self.typeck_results.node_args_mut().insert(hir_id, args);
        }
    }

    #[instrument(skip(self, span), level = "debug")]
    fn visit_adjustments(&mut self, span: Span, hir_id: HirId) {
        let adjustment = self.fcx.typeck_results.borrow_mut().adjustments_mut().remove(hir_id);
        match adjustment {
            None => {
                debug!("no adjustments for node");
            }

            Some(adjustment) => {
                let resolved_adjustment = self.resolve(adjustment, &span);
                debug!(?resolved_adjustment);
                self.typeck_results.adjustments_mut().insert(hir_id, resolved_adjustment);
            }
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_rust_2024_migration_desugared_pats(&mut self, hir_id: hir::HirId) {
        if let Some(is_hard_error) = self
            .fcx
            .typeck_results
            .borrow_mut()
            .rust_2024_migration_desugared_pats_mut()
            .remove(hir_id)
        {
            debug!(
                "node is a pat whose match ergonomics are desugared by the Rust 2024 migration lint"
            );
            self.typeck_results
                .rust_2024_migration_desugared_pats_mut()
                .insert(hir_id, is_hard_error);
        }
    }

    #[instrument(skip(self, span), level = "debug")]
    fn visit_pat_adjustments(&mut self, span: Span, hir_id: HirId) {
        let adjustment = self.fcx.typeck_results.borrow_mut().pat_adjustments_mut().remove(hir_id);
        match adjustment {
            None => {
                debug!("no pat_adjustments for node");
            }

            Some(adjustment) => {
                let resolved_adjustment = self.resolve(adjustment, &span);
                debug!(?resolved_adjustment);
                self.typeck_results.pat_adjustments_mut().insert(hir_id, resolved_adjustment);
            }
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_skipped_ref_pats(&mut self, hir_id: hir::HirId) {
        if self.fcx.typeck_results.borrow_mut().skipped_ref_pats_mut().remove(hir_id) {
            debug!("node is a skipped ref pat");
            self.typeck_results.skipped_ref_pats_mut().insert(hir_id);
        }
    }

    fn visit_liberated_fn_sigs(&mut self) {
        let fcx_typeck_results = self.fcx.typeck_results.borrow();
        assert_eq!(fcx_typeck_results.hir_owner, self.typeck_results.hir_owner);
        let common_hir_owner = fcx_typeck_results.hir_owner;

        let fcx_liberated_fn_sigs = fcx_typeck_results.liberated_fn_sigs().items_in_stable_order();

        for (local_id, &fn_sig) in fcx_liberated_fn_sigs {
            let hir_id = HirId { owner: common_hir_owner, local_id };
            let fn_sig = self.resolve(fn_sig, &hir_id);
            self.typeck_results.liberated_fn_sigs_mut().insert(hir_id, fn_sig);
        }
    }

    fn visit_fru_field_types(&mut self) {
        let fcx_typeck_results = self.fcx.typeck_results.borrow();
        assert_eq!(fcx_typeck_results.hir_owner, self.typeck_results.hir_owner);
        let common_hir_owner = fcx_typeck_results.hir_owner;

        let fcx_fru_field_types = fcx_typeck_results.fru_field_types().items_in_stable_order();

        for (local_id, ftys) in fcx_fru_field_types {
            let hir_id = HirId { owner: common_hir_owner, local_id };
            let ftys = self.resolve(ftys.clone(), &hir_id);
            self.typeck_results.fru_field_types_mut().insert(hir_id, ftys);
        }
    }

    fn visit_offset_of_container_types(&mut self) {
        let fcx_typeck_results = self.fcx.typeck_results.borrow();
        assert_eq!(fcx_typeck_results.hir_owner, self.typeck_results.hir_owner);
        let common_hir_owner = fcx_typeck_results.hir_owner;

        for (local_id, &(container, ref indices)) in
            fcx_typeck_results.offset_of_data().items_in_stable_order()
        {
            let hir_id = HirId { owner: common_hir_owner, local_id };
            let container = self.resolve(container, &hir_id);
            self.typeck_results.offset_of_data_mut().insert(hir_id, (container, indices.clone()));
        }
    }

    fn resolve<T>(&mut self, value: T, span: &dyn Locatable) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let value = self.fcx.resolve_vars_if_possible(value);
        let value = value.fold_with(&mut Resolver::new(self.fcx, span, self.body));
        assert!(!value.has_infer());

        // We may have introduced e.g. `ty::Error`, if inference failed, make sure
        // to mark the `TypeckResults` as tainted in that case, so that downstream
        // users of the typeck results don't produce extra errors, or worse, ICEs.
        if let Err(guar) = value.error_reported() {
            self.typeck_results.tainted_by_errors = Some(guar);
        }

        value
    }
}

pub(crate) trait Locatable {
    fn to_span(&self, tcx: TyCtxt<'_>) -> Span;
}

impl Locatable for Span {
    fn to_span(&self, _: TyCtxt<'_>) -> Span {
        *self
    }
}

impl Locatable for HirId {
    fn to_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.hir().span(*self)
    }
}

struct Resolver<'cx, 'tcx> {
    fcx: &'cx FnCtxt<'cx, 'tcx>,
    span: &'cx dyn Locatable,
    body: &'tcx hir::Body<'tcx>,
    /// Whether we should normalize using the new solver, disabled
    /// both when using the old solver and when resolving predicates.
    should_normalize: bool,
}

impl<'cx, 'tcx> Resolver<'cx, 'tcx> {
    fn new(
        fcx: &'cx FnCtxt<'cx, 'tcx>,
        span: &'cx dyn Locatable,
        body: &'tcx hir::Body<'tcx>,
    ) -> Resolver<'cx, 'tcx> {
        Resolver { fcx, span, body, should_normalize: fcx.next_trait_solver() }
    }

    fn report_error(&self, p: impl Into<ty::GenericArg<'tcx>>) -> ErrorGuaranteed {
        if let Some(guar) = self.fcx.tainted_by_errors() {
            guar
        } else {
            self.fcx
                .err_ctxt()
                .emit_inference_failure_err(
                    self.fcx.tcx.hir().body_owner_def_id(self.body.id()),
                    self.span.to_span(self.fcx.tcx),
                    p.into(),
                    TypeAnnotationNeeded::E0282,
                    false,
                )
                .emit()
        }
    }

    fn handle_term<T>(
        &mut self,
        value: T,
        outer_exclusive_binder: impl FnOnce(T) -> ty::DebruijnIndex,
        new_err: impl Fn(TyCtxt<'tcx>, ErrorGuaranteed) -> T,
    ) -> T
    where
        T: Into<ty::GenericArg<'tcx>> + TypeSuperFoldable<TyCtxt<'tcx>> + Copy,
    {
        let tcx = self.fcx.tcx;
        // We must deeply normalize in the new solver, since later lints
        // expect that types that show up in the typeck are fully
        // normalized.
        let mut value = if self.should_normalize {
            let body_id = tcx.hir().body_owner_def_id(self.body.id());
            let cause = ObligationCause::misc(self.span.to_span(tcx), body_id);
            let at = self.fcx.at(&cause, self.fcx.param_env);
            let universes = vec![None; outer_exclusive_binder(value).as_usize()];
            solve::deeply_normalize_with_skipped_universes(at, value, universes).unwrap_or_else(
                |errors| {
                    let guar = self.fcx.err_ctxt().report_fulfillment_errors(errors);
                    new_err(tcx, guar)
                },
            )
        } else {
            value
        };

        // Bail if there are any non-region infer.
        if value.has_non_region_infer() {
            let guar = self.report_error(value);
            value = new_err(tcx, guar);
        }

        // Erase the regions from the ty, since it's not really meaningful what
        // these region values are; there's not a trivial correspondence between
        // regions in the HIR and MIR, so when we turn the body into MIR, there's
        // no reason to keep regions around. They will be repopulated during MIR
        // borrowck, and specifically region constraints will be populated during
        // MIR typeck which is run on the new body.
        //
        // We're not using `tcx.erase_regions` as that also anonymizes bound variables,
        // regressing borrowck diagnostics.
        value = fold_regions(tcx, value, |_, _| tcx.lifetimes.re_erased);

        // Normalize consts in writeback, because GCE doesn't normalize eagerly.
        if tcx.features().generic_const_exprs() {
            value = value.fold_with(&mut EagerlyNormalizeConsts::new(self.fcx));
        }

        value
    }
}

impl<'cx, 'tcx> TypeFolder<TyCtxt<'tcx>> for Resolver<'cx, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.fcx.tcx
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        debug_assert!(!r.is_bound(), "Should not be resolving bound region.");
        self.fcx.tcx.lifetimes.re_erased
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.handle_term(ty, Ty::outer_exclusive_binder, Ty::new_error)
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        self.handle_term(ct, ty::Const::outer_exclusive_binder, |tcx, guar| {
            ty::Const::new_error(tcx, guar)
        })
        .super_fold_with(self)
    }

    fn fold_predicate(&mut self, predicate: ty::Predicate<'tcx>) -> ty::Predicate<'tcx> {
        // Do not normalize predicates in the new solver. The new solver is
        // supposed to handle unnormalized predicates and incorrectly normalizing
        // them can be unsound, e.g. for `WellFormed` predicates.
        let prev = mem::replace(&mut self.should_normalize, false);
        let predicate = predicate.super_fold_with(self);
        self.should_normalize = prev;
        predicate
    }
}

struct EagerlyNormalizeConsts<'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
}
impl<'tcx> EagerlyNormalizeConsts<'tcx> {
    fn new(fcx: &FnCtxt<'_, 'tcx>) -> Self {
        // FIXME(#132279, generic_const_exprs): Using `try_normalize_erasing_regions` here
        // means we can't handle opaque types in their defining scope.
        EagerlyNormalizeConsts { tcx: fcx.tcx, typing_env: fcx.typing_env(fcx.param_env) }
    }
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for EagerlyNormalizeConsts<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        self.tcx.try_normalize_erasing_regions(self.typing_env, ct).unwrap_or(ct)
    }
}
