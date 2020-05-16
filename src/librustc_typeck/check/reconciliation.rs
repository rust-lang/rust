use crate::check::{FnCtxt, Needs, PlaceOp};
use rustc_hir as hir;
use rustc_middle::ty::adjustment::{Adjust, Adjustment, OverloadedDeref, PointerCast};
use rustc_middle::ty::adjustment::{AllowTwoPhase, AutoBorrow, AutoBorrowMutability};
use rustc_middle::ty::{self, Ty};

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    /// Convert auto-derefs, indices, etc of an expression from `Deref` and `Index`
    /// into `DerefMut` and `IndexMut` respectively.
    ///
    /// This is a second pass of typechecking derefs/indices. We need this we do not
    /// always know whether a place needs to be mutable or not in the first pass.
    /// This happens whether there is an implicit mutable reborrow, e.g. when the type
    /// is used as the receiver of a method call.
    pub fn convert_place_derefs_to_mutable(&self, expr: &hir::Expr<'_>) {
        // Gather up expressions we want to munge.
        let mut exprs = vec![expr];

        loop {
            match exprs.last().unwrap().kind {
                hir::ExprKind::Field(ref expr, _)
                | hir::ExprKind::Index(ref expr, _)
                | hir::ExprKind::Unary(hir::UnOp::UnDeref, ref expr) => exprs.push(&expr),
                _ => break,
            }
        }

        debug!("convert_place_derefs_to_mutable: exprs={:?}", exprs);

        // Fix up autoderefs and derefs.
        for (i, &expr) in exprs.iter().rev().enumerate() {
            debug!("convert_place_derefs_to_mutable: i={} expr={:?}", i, expr);

            // Fix up the autoderefs. Autorefs can only occur immediately preceding
            // overloaded place ops, and will be fixed by them in order to get
            // the correct region.
            let mut source = self.node_ty(expr.hir_id);
            // Do not mutate adjustments in place, but rather take them,
            // and replace them after mutating them, to avoid having the
            // tables borrowed during (`deref_mut`) method resolution.
            let previous_adjustments =
                self.tables.borrow_mut().adjustments_mut().remove(expr.hir_id);
            if let Some(mut adjustments) = previous_adjustments {
                let needs = Needs::MutPlace;
                for adjustment in &mut adjustments {
                    if let Adjust::Deref(Some(ref mut deref)) = adjustment.kind {
                        if let Some(ok) = self.try_overloaded_deref(expr.span, source, needs) {
                            let method = self.register_infer_ok_obligations(ok);
                            if let ty::Ref(region, _, mutbl) = method.sig.output().kind {
                                *deref = OverloadedDeref { region, mutbl };
                            }
                        }
                    }
                    source = adjustment.target;
                }
                self.tables.borrow_mut().adjustments_mut().insert(expr.hir_id, adjustments);
            }

            match expr.kind {
                hir::ExprKind::Index(ref base_expr, ref index_expr) => {
                    // We need to get the final type in case dereferences were needed for the trait
                    // to apply (#72002).
                    let index_expr_ty = self.tables.borrow().expr_ty_adjusted(index_expr);
                    self.convert_place_op_to_mutable(
                        PlaceOp::Index,
                        expr,
                        base_expr,
                        &[index_expr_ty],
                    );
                }
                hir::ExprKind::Unary(hir::UnOp::UnDeref, ref base_expr) => {
                    self.convert_place_op_to_mutable(PlaceOp::Deref, expr, base_expr, &[]);
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
        arg_tys: &[Ty<'tcx>],
    ) {
        debug!("convert_place_op_to_mutable({:?}, {:?}, {:?}, {:?})", op, expr, base_expr, arg_tys);
        if !self.tables.borrow().is_method_call(expr) {
            debug!("convert_place_op_to_mutable - builtin, nothing to do");
            return;
        }

        let base_ty = self
            .tables
            .borrow()
            .expr_adjustments(base_expr)
            .last()
            .map_or_else(|| self.node_ty(expr.hir_id), |adj| adj.target);
        let base_ty = self.resolve_vars_if_possible(&base_ty);

        // Need to deref because overloaded place ops take self by-reference.
        let base_ty =
            base_ty.builtin_deref(false).expect("place op takes something that is not a ref").ty;

        let method = self.try_overloaded_place_op(expr.span, base_ty, arg_tys, Needs::MutPlace, op);
        let method = match method {
            Some(ok) => self.register_infer_ok_obligations(ok),
            None => return self.tcx.sess.delay_span_bug(expr.span, "re-trying op failed"),
        };
        debug!("convert_place_op_to_mutable: method={:?}", method);
        self.write_method_call(expr.hir_id, method);

        let (region, mutbl) = if let ty::Ref(r, _, mutbl) = method.sig.inputs()[0].kind {
            (r, mutbl)
        } else {
            span_bug!(expr.span, "input to place op is not a ref?");
        };

        // Convert the autoref in the base expr to mutable with the correct
        // region and mutability.
        let base_expr_ty = self.node_ty(base_expr.hir_id);
        if let Some(adjustments) =
            self.tables.borrow_mut().adjustments_mut().get_mut(base_expr.hir_id)
        {
            let mut source = base_expr_ty;
            for adjustment in &mut adjustments[..] {
                if let Adjust::Borrow(AutoBorrow::Ref(..)) = adjustment.kind {
                    debug!("convert_place_op_to_mutable: converting autoref {:?}", adjustment);
                    let mutbl = match mutbl {
                        hir::Mutability::Not => AutoBorrowMutability::Not,
                        hir::Mutability::Mut => AutoBorrowMutability::Mut {
                            // For initial two-phase borrow
                            // deployment, conservatively omit
                            // overloaded operators.
                            allow_two_phase_borrow: AllowTwoPhase::No,
                        },
                    };
                    adjustment.kind = Adjust::Borrow(AutoBorrow::Ref(region, mutbl));
                    adjustment.target =
                        self.tcx.mk_ref(region, ty::TypeAndMut { ty: source, mutbl: mutbl.into() });
                }
                source = adjustment.target;
            }

            // If we have an autoref followed by unsizing at the end, fix the unsize target.

            if let [.., Adjustment { kind: Adjust::Borrow(AutoBorrow::Ref(..)), .. }, Adjustment { kind: Adjust::Pointer(PointerCast::Unsize), ref mut target }] =
                adjustments[..]
            {
                *target = method.sig.inputs()[0];
            }
        }
    }
}
