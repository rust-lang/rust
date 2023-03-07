use crate::method::MethodCallee;
use crate::{has_expected_num_generic_args, FnCtxt, PlaceOp};
use rustc_ast as ast;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir_analysis::autoderef::Autoderef;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::InferOk;
use rustc_middle::ty::adjustment::{Adjust, Adjustment, OverloadedDeref, PointerCast};
use rustc_middle::ty::adjustment::{AllowTwoPhase, AutoBorrow, AutoBorrowMutability};
use rustc_middle::ty::{self, Ty};
use rustc_span::symbol::{sym, Ident};
use rustc_span::Span;

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    /// Type-check `*oprnd_expr` with `oprnd_expr` type-checked already.
    pub(super) fn lookup_derefing(
        &self,
        expr: &hir::Expr<'_>,
        oprnd_expr: &'tcx hir::Expr<'tcx>,
        oprnd_ty: Ty<'tcx>,
    ) -> Option<Ty<'tcx>> {
        if let Some(mt) = oprnd_ty.builtin_deref(true) {
            return Some(mt.ty);
        }

        let ok = self.try_overloaded_deref(expr.span, oprnd_ty)?;
        let method = self.register_infer_ok_obligations(ok);
        if let ty::Ref(region, _, hir::Mutability::Not) = method.sig.inputs()[0].kind() {
            self.apply_adjustments(
                oprnd_expr,
                vec![Adjustment {
                    kind: Adjust::Borrow(AutoBorrow::Ref(*region, AutoBorrowMutability::Not)),
                    target: method.sig.inputs()[0],
                }],
            );
        } else {
            span_bug!(expr.span, "input to deref is not a ref?");
        }
        let ty = self.make_overloaded_place_return_type(method).ty;
        self.write_method_call(expr.hir_id, method);
        Some(ty)
    }

    /// Type-check `*base_expr[index_expr]` with `base_expr` and `index_expr` type-checked already.
    pub(super) fn lookup_indexing(
        &self,
        expr: &hir::Expr<'_>,
        base_expr: &'tcx hir::Expr<'tcx>,
        base_ty: Ty<'tcx>,
        index_expr: &'tcx hir::Expr<'tcx>,
        idx_ty: Ty<'tcx>,
    ) -> Option<(/*index type*/ Ty<'tcx>, /*element type*/ Ty<'tcx>)> {
        // FIXME(#18741) -- this is almost but not quite the same as the
        // autoderef that normal method probing does. They could likely be
        // consolidated.

        let mut autoderef = self.autoderef(base_expr.span, base_ty);
        let mut result = None;
        while result.is_none() && autoderef.next().is_some() {
            result = self.try_index_step(expr, base_expr, &autoderef, idx_ty, index_expr);
        }
        self.register_predicates(autoderef.into_obligations());
        result
    }

    fn negative_index(
        &self,
        ty: Ty<'tcx>,
        span: Span,
        base_expr: &hir::Expr<'_>,
    ) -> Option<(Ty<'tcx>, Ty<'tcx>)> {
        let ty = self.resolve_vars_if_possible(ty);
        let mut err = self.tcx.sess.struct_span_err(
            span,
            &format!("negative integers cannot be used to index on a `{ty}`"),
        );
        err.span_label(span, &format!("cannot use a negative integer for indexing on `{ty}`"));
        if let (hir::ExprKind::Path(..), Ok(snippet)) =
            (&base_expr.kind, self.tcx.sess.source_map().span_to_snippet(base_expr.span))
        {
            // `foo[-1]` to `foo[foo.len() - 1]`
            err.span_suggestion_verbose(
                span.shrink_to_lo(),
                &format!(
                    "to access an element starting from the end of the `{ty}`, compute the index",
                ),
                format!("{snippet}.len() "),
                Applicability::MachineApplicable,
            );
        }
        let reported = err.emit();
        Some((self.tcx.ty_error(reported), self.tcx.ty_error(reported)))
    }

    /// To type-check `base_expr[index_expr]`, we progressively autoderef
    /// (and otherwise adjust) `base_expr`, looking for a type which either
    /// supports builtin indexing or overloaded indexing.
    /// This loop implements one step in that search; the autoderef loop
    /// is implemented by `lookup_indexing`.
    fn try_index_step(
        &self,
        expr: &hir::Expr<'_>,
        base_expr: &hir::Expr<'_>,
        autoderef: &Autoderef<'a, 'tcx>,
        index_ty: Ty<'tcx>,
        index_expr: &hir::Expr<'_>,
    ) -> Option<(/*index type*/ Ty<'tcx>, /*element type*/ Ty<'tcx>)> {
        let adjusted_ty =
            self.structurally_resolved_type(autoderef.span(), autoderef.final_ty(false));
        debug!(
            "try_index_step(expr={:?}, base_expr={:?}, adjusted_ty={:?}, \
             index_ty={:?})",
            expr, base_expr, adjusted_ty, index_ty
        );

        if let hir::ExprKind::Unary(
            hir::UnOp::Neg,
            hir::Expr {
                kind: hir::ExprKind::Lit(hir::Lit { node: ast::LitKind::Int(..), .. }),
                ..
            },
        ) = index_expr.kind
        {
            match adjusted_ty.kind() {
                ty::Adt(def, _) if self.tcx.is_diagnostic_item(sym::Vec, def.did()) => {
                    return self.negative_index(adjusted_ty, index_expr.span, base_expr);
                }
                ty::Slice(_) | ty::Array(_, _) => {
                    return self.negative_index(adjusted_ty, index_expr.span, base_expr);
                }
                _ => {}
            }
        }

        for unsize in [false, true] {
            let mut self_ty = adjusted_ty;
            if unsize {
                // We only unsize arrays here.
                if let ty::Array(element_ty, _) = adjusted_ty.kind() {
                    self_ty = self.tcx.mk_slice(*element_ty);
                } else {
                    continue;
                }
            }

            // If some lookup succeeds, write callee into table and extract index/element
            // type from the method signature.
            // If some lookup succeeded, install method in table
            let input_ty = self.next_ty_var(TypeVariableOrigin {
                kind: TypeVariableOriginKind::AutoDeref,
                span: base_expr.span,
            });
            let method =
                self.try_overloaded_place_op(expr.span, self_ty, &[input_ty], PlaceOp::Index);

            if let Some(result) = method {
                debug!("try_index_step: success, using overloaded indexing");
                let method = self.register_infer_ok_obligations(result);

                let mut adjustments = self.adjust_steps(autoderef);
                if let ty::Ref(region, _, hir::Mutability::Not) = method.sig.inputs()[0].kind() {
                    adjustments.push(Adjustment {
                        kind: Adjust::Borrow(AutoBorrow::Ref(*region, AutoBorrowMutability::Not)),
                        target: self.tcx.mk_ref(
                            *region,
                            ty::TypeAndMut { mutbl: hir::Mutability::Not, ty: adjusted_ty },
                        ),
                    });
                } else {
                    span_bug!(expr.span, "input to index is not a ref?");
                }
                if unsize {
                    adjustments.push(Adjustment {
                        kind: Adjust::Pointer(PointerCast::Unsize),
                        target: method.sig.inputs()[0],
                    });
                }
                self.apply_adjustments(base_expr, adjustments);

                self.write_method_call(expr.hir_id, method);

                return Some((input_ty, self.make_overloaded_place_return_type(method).ty));
            }
        }

        None
    }

    /// Try to resolve an overloaded place op. We only deal with the immutable
    /// variant here (Deref/Index). In some contexts we would need the mutable
    /// variant (DerefMut/IndexMut); those would be later converted by
    /// `convert_place_derefs_to_mutable`.
    pub(super) fn try_overloaded_place_op(
        &self,
        span: Span,
        base_ty: Ty<'tcx>,
        arg_tys: &[Ty<'tcx>],
        op: PlaceOp,
    ) -> Option<InferOk<'tcx, MethodCallee<'tcx>>> {
        debug!("try_overloaded_place_op({:?},{:?},{:?})", span, base_ty, op);

        let (imm_tr, imm_op) = match op {
            PlaceOp::Deref => (self.tcx.lang_items().deref_trait(), sym::deref),
            PlaceOp::Index => (self.tcx.lang_items().index_trait(), sym::index),
        };

        // If the lang item was declared incorrectly, stop here so that we don't
        // run into an ICE (#83893). The error is reported where the lang item is
        // declared.
        if !has_expected_num_generic_args(
            self.tcx,
            imm_tr,
            match op {
                PlaceOp::Deref => 0,
                PlaceOp::Index => 1,
            },
        ) {
            return None;
        }

        imm_tr.and_then(|trait_did| {
            self.lookup_method_in_trait(
                self.misc(span),
                Ident::with_dummy_span(imm_op),
                trait_did,
                base_ty,
                Some(arg_tys),
            )
        })
    }

    fn try_mutable_overloaded_place_op(
        &self,
        span: Span,
        base_ty: Ty<'tcx>,
        arg_tys: &[Ty<'tcx>],
        op: PlaceOp,
    ) -> Option<InferOk<'tcx, MethodCallee<'tcx>>> {
        debug!("try_mutable_overloaded_place_op({:?},{:?},{:?})", span, base_ty, op);

        let (mut_tr, mut_op) = match op {
            PlaceOp::Deref => (self.tcx.lang_items().deref_mut_trait(), sym::deref_mut),
            PlaceOp::Index => (self.tcx.lang_items().index_mut_trait(), sym::index_mut),
        };

        // If the lang item was declared incorrectly, stop here so that we don't
        // run into an ICE (#83893). The error is reported where the lang item is
        // declared.
        if !has_expected_num_generic_args(
            self.tcx,
            mut_tr,
            match op {
                PlaceOp::Deref => 0,
                PlaceOp::Index => 1,
            },
        ) {
            return None;
        }

        mut_tr.and_then(|trait_did| {
            self.lookup_method_in_trait(
                self.misc(span),
                Ident::with_dummy_span(mut_op),
                trait_did,
                base_ty,
                Some(arg_tys),
            )
        })
    }

    /// Convert auto-derefs, indices, etc of an expression from `Deref` and `Index`
    /// into `DerefMut` and `IndexMut` respectively.
    ///
    /// This is a second pass of typechecking derefs/indices. We need this because we do not
    /// always know whether a place needs to be mutable or not in the first pass.
    /// This happens whether there is an implicit mutable reborrow, e.g. when the type
    /// is used as the receiver of a method call.
    pub fn convert_place_derefs_to_mutable(&self, expr: &hir::Expr<'_>) {
        // Gather up expressions we want to munge.
        let mut exprs = vec![expr];

        while let hir::ExprKind::Field(ref expr, _)
        | hir::ExprKind::Index(ref expr, _)
        | hir::ExprKind::Unary(hir::UnOp::Deref, ref expr) = exprs.last().unwrap().kind
        {
            exprs.push(expr);
        }

        debug!("convert_place_derefs_to_mutable: exprs={:?}", exprs);

        // Fix up autoderefs and derefs.
        let mut inside_union = false;
        for (i, &expr) in exprs.iter().rev().enumerate() {
            debug!("convert_place_derefs_to_mutable: i={} expr={:?}", i, expr);

            let mut source = self.node_ty(expr.hir_id);
            if matches!(expr.kind, hir::ExprKind::Unary(hir::UnOp::Deref, _)) {
                // Clear previous flag; after a pointer indirection it does not apply any more.
                inside_union = false;
            }
            if source.is_union() {
                inside_union = true;
            }
            // Fix up the autoderefs. Autorefs can only occur immediately preceding
            // overloaded place ops, and will be fixed by them in order to get
            // the correct region.
            // Do not mutate adjustments in place, but rather take them,
            // and replace them after mutating them, to avoid having the
            // typeck results borrowed during (`deref_mut`) method resolution.
            let previous_adjustments =
                self.typeck_results.borrow_mut().adjustments_mut().remove(expr.hir_id);
            if let Some(mut adjustments) = previous_adjustments {
                for adjustment in &mut adjustments {
                    if let Adjust::Deref(Some(ref mut deref)) = adjustment.kind
                        && let Some(ok) = self.try_mutable_overloaded_place_op(
                            expr.span,
                            source,
                            &[],
                            PlaceOp::Deref,
                        )
                    {
                        let method = self.register_infer_ok_obligations(ok);
                        if let ty::Ref(region, _, mutbl) = *method.sig.output().kind() {
                            *deref = OverloadedDeref { region, mutbl, span: deref.span };
                        }
                        // If this is a union field, also throw an error for `DerefMut` of `ManuallyDrop` (see RFC 2514).
                        // This helps avoid accidental drops.
                        if inside_union
                            && source.ty_adt_def().map_or(false, |adt| adt.is_manually_drop())
                        {
                            let mut err = self.tcx.sess.struct_span_err(
                                expr.span,
                                "not automatically applying `DerefMut` on `ManuallyDrop` union field",
                            );
                            err.help(
                                "writing to this reference calls the destructor for the old value",
                            );
                            err.help("add an explicit `*` if that is desired, or call `ptr::write` to not run the destructor");
                            err.emit();
                        }
                    }
                    source = adjustment.target;
                }
                self.typeck_results.borrow_mut().adjustments_mut().insert(expr.hir_id, adjustments);
            }

            match expr.kind {
                hir::ExprKind::Index(base_expr, ..) => {
                    self.convert_place_op_to_mutable(PlaceOp::Index, expr, base_expr);
                }
                hir::ExprKind::Unary(hir::UnOp::Deref, base_expr) => {
                    self.convert_place_op_to_mutable(PlaceOp::Deref, expr, base_expr);
                }
                _ => {}
            }
        }
    }

    fn convert_place_op_to_mutable(
        &self,
        op: PlaceOp,
        expr: &hir::Expr<'_>,
        base_expr: &hir::Expr<'_>,
    ) {
        debug!("convert_place_op_to_mutable({:?}, {:?}, {:?})", op, expr, base_expr);
        if !self.typeck_results.borrow().is_method_call(expr) {
            debug!("convert_place_op_to_mutable - builtin, nothing to do");
            return;
        }

        // Need to deref because overloaded place ops take self by-reference.
        let base_ty = self
            .typeck_results
            .borrow()
            .expr_ty_adjusted(base_expr)
            .builtin_deref(false)
            .expect("place op takes something that is not a ref")
            .ty;

        let arg_ty = match op {
            PlaceOp::Deref => None,
            PlaceOp::Index => {
                // We would need to recover the `T` used when we resolve `<_ as Index<T>>::index`
                // in try_index_step. This is the subst at index 1.
                //
                // Note: we should *not* use `expr_ty` of index_expr here because autoderef
                // during coercions can cause type of index_expr to differ from `T` (#72002).
                // We also could not use `expr_ty_adjusted` of index_expr because reborrowing
                // during coercions can also cause type of index_expr to differ from `T`,
                // which can potentially cause regionck failure (#74933).
                Some(self.typeck_results.borrow().node_substs(expr.hir_id).type_at(1))
            }
        };
        let arg_tys = arg_ty.as_slice();
        let method = self.try_mutable_overloaded_place_op(expr.span, base_ty, arg_tys, op);
        let method = match method {
            Some(ok) => self.register_infer_ok_obligations(ok),
            // Couldn't find the mutable variant of the place op, keep the
            // current, immutable version.
            None => return,
        };
        debug!("convert_place_op_to_mutable: method={:?}", method);
        self.write_method_call(expr.hir_id, method);

        let ty::Ref(region, _, hir::Mutability::Mut) = method.sig.inputs()[0].kind() else {
            span_bug!(expr.span, "input to mutable place op is not a mut ref?");
        };

        // Convert the autoref in the base expr to mutable with the correct
        // region and mutability.
        let base_expr_ty = self.node_ty(base_expr.hir_id);
        if let Some(adjustments) =
            self.typeck_results.borrow_mut().adjustments_mut().get_mut(base_expr.hir_id)
        {
            let mut source = base_expr_ty;
            for adjustment in &mut adjustments[..] {
                if let Adjust::Borrow(AutoBorrow::Ref(..)) = adjustment.kind {
                    debug!("convert_place_op_to_mutable: converting autoref {:?}", adjustment);
                    let mutbl = AutoBorrowMutability::Mut {
                        // Deref/indexing can be desugared to a method call,
                        // so maybe we could use two-phase here.
                        // See the documentation of AllowTwoPhase for why that's
                        // not the case today.
                        allow_two_phase_borrow: AllowTwoPhase::No,
                    };
                    adjustment.kind = Adjust::Borrow(AutoBorrow::Ref(*region, mutbl));
                    adjustment.target = self
                        .tcx
                        .mk_ref(*region, ty::TypeAndMut { ty: source, mutbl: mutbl.into() });
                }
                source = adjustment.target;
            }

            // If we have an autoref followed by unsizing at the end, fix the unsize target.
            if let [
                ..,
                Adjustment { kind: Adjust::Borrow(AutoBorrow::Ref(..)), .. },
                Adjustment { kind: Adjust::Pointer(PointerCast::Unsize), ref mut target },
            ] = adjustments[..]
            {
                *target = method.sig.inputs()[0];
            }
        }
    }
}
