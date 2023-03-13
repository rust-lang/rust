// Type resolution: the phase that finds all the types in the AST with
// unresolved type variables and replaces "ty_var" types with their
// substitutions.

use crate::FnCtxt;
use hir::def_id::LocalDefId;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_hir::intravisit::{self, Visitor};
use rustc_infer::infer::error_reporting::TypeAnnotationNeeded::E0282;
use rustc_infer::infer::InferCtxt;
use rustc_middle::hir::place::Place as HirPlace;
use rustc_middle::mir::FakeReadCause;
use rustc_middle::ty::adjustment::{Adjust, Adjustment, PointerCast};
use rustc_middle::ty::fold::{TypeFoldable, TypeFolder, TypeSuperFoldable};
use rustc_middle::ty::visit::{TypeSuperVisitable, TypeVisitable, TypeVisitableExt};
use rustc_middle::ty::TypeckResults;
use rustc_middle::ty::{self, ClosureSizeProfileData, Ty, TyCtxt};
use rustc_span::symbol::sym;
use rustc_span::Span;

use std::mem;
use std::ops::ControlFlow;

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
    pub fn resolve_type_vars_in_body(
        &self,
        body: &'tcx hir::Body<'tcx>,
    ) -> &'tcx ty::TypeckResults<'tcx> {
        let item_def_id = self.tcx.hir().body_owner_def_id(body.id());

        // This attribute causes us to dump some writeback information
        // in the form of errors, which is used for unit tests.
        let rustc_dump_user_substs = self.tcx.has_attr(item_def_id, sym::rustc_dump_user_substs);

        let mut wbcx = WritebackCx::new(self, body, rustc_dump_user_substs);
        for param in body.params {
            wbcx.visit_node_id(param.pat.span, param.hir_id);
        }
        // Type only exists for constants and statics, not functions.
        match self.tcx.hir().body_owner_kind(item_def_id) {
            hir::BodyOwnerKind::Const | hir::BodyOwnerKind::Static(_) => {
                let item_hir_id = self.tcx.hir().local_def_id_to_hir_id(item_def_id);
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
        wbcx.visit_generator_interior_types();

        wbcx.typeck_results.rvalue_scopes =
            mem::take(&mut self.typeck_results.borrow_mut().rvalue_scopes);

        let used_trait_imports =
            mem::take(&mut self.typeck_results.borrow_mut().used_trait_imports);
        debug!("used_trait_imports({:?}) = {:?}", item_def_id, used_trait_imports);
        wbcx.typeck_results.used_trait_imports = used_trait_imports;

        wbcx.typeck_results.treat_byte_string_as_slice =
            mem::take(&mut self.typeck_results.borrow_mut().treat_byte_string_as_slice);

        if let Some(e) = self.tainted_by_errors() {
            wbcx.typeck_results.tainted_by_errors = Some(e);
        }

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

    rustc_dump_user_substs: bool,
}

impl<'cx, 'tcx> WritebackCx<'cx, 'tcx> {
    fn new(
        fcx: &'cx FnCtxt<'cx, 'tcx>,
        body: &'tcx hir::Body<'tcx>,
        rustc_dump_user_substs: bool,
    ) -> WritebackCx<'cx, 'tcx> {
        let owner = body.id().hir_id.owner;

        WritebackCx {
            fcx,
            typeck_results: ty::TypeckResults::new(owner),
            body,
            rustc_dump_user_substs,
        }
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.fcx.tcx
    }

    fn write_ty_to_typeck_results(&mut self, hir_id: hir::HirId, ty: Ty<'tcx>) {
        debug!("write_ty_to_typeck_results({:?}, {:?})", hir_id, ty);
        assert!(!ty.needs_infer() && !ty.has_placeholders() && !ty.has_free_regions());
        self.typeck_results.node_types_mut().insert(hir_id, ty);
    }

    // Hacky hack: During type-checking, we treat *all* operators
    // as potentially overloaded. But then, during writeback, if
    // we observe that something like `a+b` is (known to be)
    // operating on scalars, we clear the overload.
    fn fix_scalar_builtin_expr(&mut self, e: &hir::Expr<'_>) {
        match e.kind {
            hir::ExprKind::Unary(hir::UnOp::Neg | hir::UnOp::Not, inner) => {
                let inner_ty = self.fcx.node_ty(inner.hir_id);
                let inner_ty = self.fcx.resolve_vars_if_possible(inner_ty);

                if inner_ty.is_scalar() {
                    let mut typeck_results = self.fcx.typeck_results.borrow_mut();
                    typeck_results.type_dependent_defs_mut().remove(e.hir_id);
                    typeck_results.node_substs_mut().remove(e.hir_id);
                }
            }
            hir::ExprKind::Binary(ref op, lhs, rhs) | hir::ExprKind::AssignOp(ref op, lhs, rhs) => {
                let lhs_ty = self.fcx.node_ty(lhs.hir_id);
                let lhs_ty = self.fcx.resolve_vars_if_possible(lhs_ty);

                let rhs_ty = self.fcx.node_ty(rhs.hir_id);
                let rhs_ty = self.fcx.resolve_vars_if_possible(rhs_ty);

                if lhs_ty.is_scalar() && rhs_ty.is_scalar() {
                    let mut typeck_results = self.fcx.typeck_results.borrow_mut();
                    typeck_results.type_dependent_defs_mut().remove(e.hir_id);
                    typeck_results.node_substs_mut().remove(e.hir_id);

                    match e.kind {
                        hir::ExprKind::Binary(..) => {
                            if !op.node.is_by_value() {
                                let mut adjustments = typeck_results.adjustments_mut();
                                if let Some(a) = adjustments.get_mut(lhs.hir_id) {
                                    a.pop();
                                }
                                if let Some(a) = adjustments.get_mut(rhs.hir_id) {
                                    a.pop();
                                }
                            }
                        }
                        hir::ExprKind::AssignOp(..)
                            if let Some(a) = typeck_results.adjustments_mut().get_mut(lhs.hir_id) =>
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
        typeck_results: &TypeckResults<'tcx>,
        e: &hir::Expr<'_>,
        base_ty: Ty<'tcx>,
        index_ty: Ty<'tcx>,
    ) -> bool {
        if let Some(elem_ty) = base_ty.builtin_index() {
            let Some(exp_ty) = typeck_results.expr_ty_opt(e) else {return false;};
            let resolved_exp_ty = self.resolve(exp_ty, &e.span);

            elem_ty == resolved_exp_ty && index_ty == self.fcx.tcx.types.usize
        } else {
            false
        }
    }

    // Similar to operators, indexing is always assumed to be overloaded
    // Here, correct cases where an indexing expression can be simplified
    // to use builtin indexing because the index type is known to be
    // usize-ish
    fn fix_index_builtin_expr(&mut self, e: &hir::Expr<'_>) {
        if let hir::ExprKind::Index(ref base, ref index) = e.kind {
            let mut typeck_results = self.fcx.typeck_results.borrow_mut();

            // All valid indexing looks like this; might encounter non-valid indexes at this point.
            let base_ty = typeck_results
                .expr_ty_adjusted_opt(base)
                .map(|t| self.fcx.resolve_vars_if_possible(t).kind());
            if base_ty.is_none() {
                // When encountering `return [0][0]` outside of a `fn` body we can encounter a base
                // that isn't in the type table. We assume more relevant errors have already been
                // emitted, so we delay an ICE if none have. (#64638)
                self.tcx().sess.delay_span_bug(e.span, &format!("bad base: `{:?}`", base));
            }
            if let Some(ty::Ref(_, base_ty, _)) = base_ty {
                let index_ty = typeck_results.expr_ty_adjusted_opt(index).unwrap_or_else(|| {
                    // When encountering `return [0][0]` outside of a `fn` body we would attempt
                    // to access an nonexistent index. We assume that more relevant errors will
                    // already have been emitted, so we only gate on this with an ICE if no
                    // error has been emitted. (#64638)
                    self.fcx.tcx.ty_error_with_message(
                        e.span,
                        &format!("bad index {:?} for base: `{:?}`", index, base),
                    )
                });
                let index_ty = self.fcx.resolve_vars_if_possible(index_ty);
                let resolved_base_ty = self.resolve(*base_ty, &base.span);

                if self.is_builtin_index(&typeck_results, e, resolved_base_ty, index_ty) {
                    // Remove the method call record
                    typeck_results.type_dependent_defs_mut().remove(e.hir_id);
                    typeck_results.node_substs_mut().remove(e.hir_id);

                    if let Some(a) = typeck_results.adjustments_mut().get_mut(base.hir_id) {
                        // Discard the need for a mutable borrow

                        // Extra adjustment made when indexing causes a drop
                        // of size information - we need to get rid of it
                        // Since this is "after" the other adjustment to be
                        // discarded, we do an extra `pop()`
                        if let Some(Adjustment {
                            kind: Adjust::Pointer(PointerCast::Unsize), ..
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
        self.fix_scalar_builtin_expr(e);
        self.fix_index_builtin_expr(e);

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
            hir::ExprKind::Field(..) => {
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
    }

    fn visit_generic_param(&mut self, p: &'tcx hir::GenericParam<'tcx>) {
        match &p.kind {
            hir::GenericParamKind::Lifetime { .. } => {
                // Nothing to write back here
            }
            hir::GenericParamKind::Type { .. } | hir::GenericParamKind::Const { .. } => {
                self.tcx().sess.delay_span_bug(p.span, format!("unexpected generic param: {p:?}"));
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

        self.visit_pat_adjustments(p.span, p.hir_id);

        self.visit_node_id(p.span, p.hir_id);
        intravisit::walk_pat(self, p);
    }

    fn visit_local(&mut self, l: &'tcx hir::Local<'tcx>) {
        intravisit::walk_local(self, l);
        let var_ty = self.fcx.local_ty(l.span, l.hir_id).decl_ty;
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
        let mut res: FxHashMap<LocalDefId, ClosureSizeProfileData<'tcx>> = Default::default();
        for (&closure_def_id, data) in self.fcx.typeck_results.borrow().closure_size_eval.iter() {
            let closure_hir_id = self.tcx().hir().local_def_id_to_hir_id(closure_def_id);

            let data = self.resolve(*data, &closure_hir_id);

            res.insert(closure_def_id, data);
        }

        self.typeck_results.closure_size_eval = res;
    }
    fn visit_min_capture_map(&mut self) {
        let mut min_captures_wb = ty::MinCaptureInformationMap::with_capacity_and_hasher(
            self.fcx.typeck_results.borrow().closure_min_captures.len(),
            Default::default(),
        );
        for (&closure_def_id, root_min_captures) in
            self.fcx.typeck_results.borrow().closure_min_captures.iter()
        {
            let mut root_var_map_wb = ty::RootVariableMinCaptureList::with_capacity_and_hasher(
                root_min_captures.len(),
                Default::default(),
            );
            for (var_hir_id, min_list) in root_min_captures.iter() {
                let min_list_wb = min_list
                    .iter()
                    .map(|captured_place| {
                        let locatable = captured_place.info.path_expr_id.unwrap_or_else(|| {
                            self.tcx().hir().local_def_id_to_hir_id(closure_def_id)
                        });

                        self.resolve(captured_place.clone(), &locatable)
                    })
                    .collect();
                root_var_map_wb.insert(*var_hir_id, min_list_wb);
            }
            min_captures_wb.insert(closure_def_id, root_var_map_wb);
        }

        self.typeck_results.closure_min_captures = min_captures_wb;
    }

    fn visit_fake_reads_map(&mut self) {
        let mut resolved_closure_fake_reads: FxHashMap<
            LocalDefId,
            Vec<(HirPlace<'tcx>, FakeReadCause, hir::HirId)>,
        > = Default::default();
        for (&closure_def_id, fake_reads) in
            self.fcx.typeck_results.borrow().closure_fake_reads.iter()
        {
            let mut resolved_fake_reads = Vec::<(HirPlace<'tcx>, FakeReadCause, hir::HirId)>::new();
            for (place, cause, hir_id) in fake_reads.iter() {
                let locatable = self.tcx().hir().local_def_id_to_hir_id(closure_def_id);

                let resolved_fake_read = self.resolve(place.clone(), &locatable);
                resolved_fake_reads.push((resolved_fake_read, *cause, *hir_id));
            }
            resolved_closure_fake_reads.insert(closure_def_id, resolved_fake_reads);
        }
        self.typeck_results.closure_fake_reads = resolved_closure_fake_reads;
    }

    fn visit_closures(&mut self) {
        let fcx_typeck_results = self.fcx.typeck_results.borrow();
        assert_eq!(fcx_typeck_results.hir_owner, self.typeck_results.hir_owner);
        let common_hir_owner = fcx_typeck_results.hir_owner;

        let fcx_closure_kind_origins =
            fcx_typeck_results.closure_kind_origins().items_in_stable_order();

        for (local_id, origin) in fcx_closure_kind_origins {
            let hir_id = hir::HirId { owner: common_hir_owner, local_id };
            let place_span = origin.0;
            let place = self.resolve(origin.1.clone(), &place_span);
            self.typeck_results.closure_kind_origins_mut().insert(hir_id, (place_span, place));
        }
    }

    fn visit_coercion_casts(&mut self) {
        let fcx_typeck_results = self.fcx.typeck_results.borrow();

        assert_eq!(fcx_typeck_results.hir_owner, self.typeck_results.hir_owner);

        let fcx_coercion_casts = fcx_typeck_results.coercion_casts().to_sorted_stable_ord();
        for local_id in fcx_coercion_casts {
            self.typeck_results.set_coercion_cast(local_id);
        }
    }

    fn visit_user_provided_tys(&mut self) {
        let fcx_typeck_results = self.fcx.typeck_results.borrow();
        assert_eq!(fcx_typeck_results.hir_owner, self.typeck_results.hir_owner);
        let common_hir_owner = fcx_typeck_results.hir_owner;

        if self.rustc_dump_user_substs {
            let sorted_user_provided_types =
                fcx_typeck_results.user_provided_types().items_in_stable_order();

            let mut errors_buffer = Vec::new();
            for (local_id, c_ty) in sorted_user_provided_types {
                let hir_id = hir::HirId { owner: common_hir_owner, local_id };

                if let ty::UserType::TypeOf(_, user_substs) = c_ty.value {
                    // This is a unit-testing mechanism.
                    let span = self.tcx().hir().span(hir_id);
                    // We need to buffer the errors in order to guarantee a consistent
                    // order when emitting them.
                    let err = self
                        .tcx()
                        .sess
                        .struct_span_err(span, &format!("user substs: {:?}", user_substs));
                    err.buffer(&mut errors_buffer);
                }
            }

            if !errors_buffer.is_empty() {
                errors_buffer.sort_by_key(|diag| diag.span.primary_span());
                for mut diag in errors_buffer {
                    self.tcx().sess.diagnostic().emit_diagnostic(&mut diag);
                }
            }
        }

        self.typeck_results.user_provided_types_mut().extend(
            fcx_typeck_results.user_provided_types().items().map(|(local_id, c_ty)| {
                let hir_id = hir::HirId { owner: common_hir_owner, local_id };

                if cfg!(debug_assertions) && c_ty.needs_infer() {
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

        self.typeck_results.user_provided_sigs.extend(
            fcx_typeck_results.user_provided_sigs.items().map(|(&def_id, c_sig)| {
                if cfg!(debug_assertions) && c_sig.needs_infer() {
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

    fn visit_generator_interior_types(&mut self) {
        let fcx_typeck_results = self.fcx.typeck_results.borrow();
        assert_eq!(fcx_typeck_results.hir_owner, self.typeck_results.hir_owner);
        self.typeck_results.generator_interior_types =
            fcx_typeck_results.generator_interior_types.clone();
        for (&expr_def_id, predicates) in fcx_typeck_results.generator_interior_predicates.iter() {
            let predicates = self.resolve(predicates.clone(), &self.fcx.tcx.def_span(expr_def_id));
            self.typeck_results.generator_interior_predicates.insert(expr_def_id, predicates);
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_opaque_types(&mut self) {
        let opaque_types = self.fcx.infcx.take_opaque_types();
        for (opaque_type_key, decl) in opaque_types {
            let hidden_type = self.resolve(decl.hidden_type, &decl.hidden_type.span);
            let opaque_type_key = self.resolve(opaque_type_key, &decl.hidden_type.span);

            struct RecursionChecker {
                def_id: LocalDefId,
            }
            impl<'tcx> ty::TypeVisitor<TyCtxt<'tcx>> for RecursionChecker {
                type BreakTy = ();
                fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
                    if let ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }) = *t.kind() {
                        if def_id == self.def_id.to_def_id() {
                            return ControlFlow::Break(());
                        }
                    }
                    t.super_visit_with(self)
                }
            }
            if hidden_type
                .visit_with(&mut RecursionChecker { def_id: opaque_type_key.def_id })
                .is_break()
            {
                continue;
            }

            let hidden_type = hidden_type.remap_generic_params_to_declaration_params(
                opaque_type_key,
                self.fcx.infcx.tcx,
                true,
            );

            self.typeck_results.concrete_opaque_types.insert(opaque_type_key.def_id, hidden_type);
        }
    }

    fn visit_field_id(&mut self, hir_id: hir::HirId) {
        if let Some(index) = self.fcx.typeck_results.borrow_mut().field_indices_mut().remove(hir_id)
        {
            self.typeck_results.field_indices_mut().insert(hir_id, index);
        }
    }

    #[instrument(skip(self, span), level = "debug")]
    fn visit_node_id(&mut self, span: Span, hir_id: hir::HirId) {
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

        // Resolve any substitutions
        if let Some(substs) = self.fcx.typeck_results.borrow().node_substs_opt(hir_id) {
            let substs = self.resolve(substs, &span);
            debug!("write_substs_to_tcx({:?}, {:?})", hir_id, substs);
            assert!(!substs.needs_infer() && !substs.has_placeholders());
            self.typeck_results.node_substs_mut().insert(hir_id, substs);
        }
    }

    #[instrument(skip(self, span), level = "debug")]
    fn visit_adjustments(&mut self, span: Span, hir_id: hir::HirId) {
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

    #[instrument(skip(self, span), level = "debug")]
    fn visit_pat_adjustments(&mut self, span: Span, hir_id: hir::HirId) {
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

    fn visit_liberated_fn_sigs(&mut self) {
        let fcx_typeck_results = self.fcx.typeck_results.borrow();
        assert_eq!(fcx_typeck_results.hir_owner, self.typeck_results.hir_owner);
        let common_hir_owner = fcx_typeck_results.hir_owner;

        let fcx_liberated_fn_sigs = fcx_typeck_results.liberated_fn_sigs().items_in_stable_order();

        for (local_id, &fn_sig) in fcx_liberated_fn_sigs {
            let hir_id = hir::HirId { owner: common_hir_owner, local_id };
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
            let hir_id = hir::HirId { owner: common_hir_owner, local_id };
            let ftys = self.resolve(ftys.clone(), &hir_id);
            self.typeck_results.fru_field_types_mut().insert(hir_id, ftys);
        }
    }

    fn resolve<T>(&mut self, x: T, span: &dyn Locatable) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let mut resolver = Resolver::new(self.fcx, span, self.body);
        let x = x.fold_with(&mut resolver);
        if cfg!(debug_assertions) && x.needs_infer() {
            span_bug!(span.to_span(self.fcx.tcx), "writeback: `{:?}` has inference variables", x);
        }

        // We may have introduced e.g. `ty::Error`, if inference failed, make sure
        // to mark the `TypeckResults` as tainted in that case, so that downstream
        // users of the typeck results don't produce extra errors, or worse, ICEs.
        if let Some(e) = resolver.replaced_with_error {
            self.typeck_results.tainted_by_errors = Some(e);
        }

        x
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

impl Locatable for hir::HirId {
    fn to_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.hir().span(*self)
    }
}

/// The Resolver. This is the type folding engine that detects
/// unresolved types and so forth.
struct Resolver<'cx, 'tcx> {
    tcx: TyCtxt<'tcx>,
    infcx: &'cx InferCtxt<'tcx>,
    span: &'cx dyn Locatable,
    body: &'tcx hir::Body<'tcx>,

    /// Set to `Some` if any `Ty` or `ty::Const` had to be replaced with an `Error`.
    replaced_with_error: Option<ErrorGuaranteed>,
}

impl<'cx, 'tcx> Resolver<'cx, 'tcx> {
    fn new(
        fcx: &'cx FnCtxt<'cx, 'tcx>,
        span: &'cx dyn Locatable,
        body: &'tcx hir::Body<'tcx>,
    ) -> Resolver<'cx, 'tcx> {
        Resolver { tcx: fcx.tcx, infcx: fcx, span, body, replaced_with_error: None }
    }

    fn report_error(&self, p: impl Into<ty::GenericArg<'tcx>>) -> ErrorGuaranteed {
        match self.tcx.sess.has_errors() {
            Some(e) => e,
            None => self
                .infcx
                .err_ctxt()
                .emit_inference_failure_err(
                    self.tcx.hir().body_owner_def_id(self.body.id()),
                    self.span.to_span(self.tcx),
                    p.into(),
                    E0282,
                    false,
                )
                .emit(),
        }
    }
}

struct EraseEarlyRegions<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for EraseEarlyRegions<'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if ty.has_type_flags(ty::TypeFlags::HAS_FREE_REGIONS) {
            ty.super_fold_with(self)
        } else {
            ty
        }
    }
    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        if r.is_late_bound() { r } else { self.tcx.lifetimes.re_erased }
    }
}

impl<'cx, 'tcx> TypeFolder<TyCtxt<'tcx>> for Resolver<'cx, 'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match self.infcx.fully_resolve(t) {
            Ok(t) => {
                // Do not anonymize late-bound regions
                // (e.g. keep `for<'a>` named `for<'a>`).
                // This allows NLL to generate error messages that
                // refer to the higher-ranked lifetime names written by the user.
                EraseEarlyRegions { tcx: self.tcx }.fold_ty(t)
            }
            Err(_) => {
                debug!("Resolver::fold_ty: input type `{:?}` not fully resolvable", t);
                let e = self.report_error(t);
                self.replaced_with_error = Some(e);
                self.interner().ty_error(e)
            }
        }
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        debug_assert!(!r.is_late_bound(), "Should not be resolving bound region.");
        self.tcx.lifetimes.re_erased
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        match self.infcx.fully_resolve(ct) {
            Ok(ct) => self.tcx.erase_regions(ct),
            Err(_) => {
                debug!("Resolver::fold_const: input const `{:?}` not fully resolvable", ct);
                let e = self.report_error(ct);
                self.replaced_with_error = Some(e);
                self.interner().const_error_with_guaranteed(ct.ty(), e)
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// During type check, we store promises with the result of trait
// lookup rather than the actual results (because the results are not
// necessarily available immediately). These routines unwind the
// promises. It is expected that we will have already reported any
// errors that may be encountered, so if the promises store an error,
// a dummy result is returned.
