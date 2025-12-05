//! Inference of *place operators*: deref and indexing (operators that create places, as opposed to values).

use hir_def::hir::ExprId;
use intern::sym;
use rustc_ast_ir::Mutability;
use rustc_type_ir::inherent::{IntoKind, Ty as _};
use tracing::debug;

use crate::{
    Adjust, Adjustment, AutoBorrow, PointerCast,
    autoderef::InferenceContextAutoderef,
    infer::{AllowTwoPhase, AutoBorrowMutability, InferenceContext, unify::InferenceTable},
    method_resolution::{MethodCallee, TreatNotYetDefinedOpaques},
    next_solver::{
        ClauseKind, Ty, TyKind,
        infer::{
            InferOk,
            traits::{Obligation, ObligationCause},
        },
    },
};

#[derive(Debug, Copy, Clone)]
pub(super) enum PlaceOp {
    Deref,
    Index,
}

impl<'a, 'db> InferenceContext<'a, 'db> {
    pub(super) fn try_overloaded_deref(
        &self,
        base_ty: Ty<'db>,
    ) -> Option<InferOk<'db, MethodCallee<'db>>> {
        self.try_overloaded_place_op(base_ty, None, PlaceOp::Deref)
    }

    /// For the overloaded place expressions (`*x`, `x[3]`), the trait
    /// returns a type of `&T`, but the actual type we assign to the
    /// *expression* is `T`. So this function just peels off the return
    /// type by one layer to yield `T`.
    fn make_overloaded_place_return_type(&self, method: MethodCallee<'db>) -> Ty<'db> {
        // extract method return type, which will be &T;
        let ret_ty = method.sig.output();

        // method returns &T, but the type as visible to user is T, so deref
        ret_ty.builtin_deref(true).unwrap()
    }

    /// Type-check `*oprnd_expr` with `oprnd_expr` type-checked already.
    pub(super) fn lookup_derefing(
        &mut self,
        expr: ExprId,
        oprnd_expr: ExprId,
        oprnd_ty: Ty<'db>,
    ) -> Option<Ty<'db>> {
        if let Some(ty) = oprnd_ty.builtin_deref(true) {
            return Some(ty);
        }

        let ok = self.try_overloaded_deref(oprnd_ty)?;
        let method = self.table.register_infer_ok(ok);
        if let TyKind::Ref(_, _, Mutability::Not) = method.sig.inputs_and_output.inputs()[0].kind()
        {
            self.write_expr_adj(
                oprnd_expr,
                Box::new([Adjustment {
                    kind: Adjust::Borrow(AutoBorrow::Ref(AutoBorrowMutability::Not)),
                    target: method.sig.inputs_and_output.inputs()[0],
                }]),
            );
        } else {
            panic!("input to deref is not a ref?");
        }
        let ty = self.make_overloaded_place_return_type(method);
        self.write_method_resolution(expr, method.def_id, method.args);
        Some(ty)
    }

    /// Type-check `*base_expr[index_expr]` with `base_expr` and `index_expr` type-checked already.
    pub(super) fn lookup_indexing(
        &mut self,
        expr: ExprId,
        base_expr: ExprId,
        base_ty: Ty<'db>,
        idx_ty: Ty<'db>,
    ) -> Option<(/*index type*/ Ty<'db>, /*element type*/ Ty<'db>)> {
        // FIXME(#18741) -- this is almost but not quite the same as the
        // autoderef that normal method probing does. They could likely be
        // consolidated.

        let mut autoderef = InferenceContextAutoderef::new_from_inference_context(self, base_ty);
        let mut result = None;
        while result.is_none() && autoderef.next().is_some() {
            result = Self::try_index_step(expr, base_expr, &mut autoderef, idx_ty);
        }
        result
    }

    /// To type-check `base_expr[index_expr]`, we progressively autoderef
    /// (and otherwise adjust) `base_expr`, looking for a type which either
    /// supports builtin indexing or overloaded indexing.
    /// This loop implements one step in that search; the autoderef loop
    /// is implemented by `lookup_indexing`.
    fn try_index_step(
        expr: ExprId,
        base_expr: ExprId,
        autoderef: &mut InferenceContextAutoderef<'_, 'a, 'db>,
        index_ty: Ty<'db>,
    ) -> Option<(/*index type*/ Ty<'db>, /*element type*/ Ty<'db>)> {
        let ty = autoderef.final_ty();
        let adjusted_ty = autoderef.ctx().table.structurally_resolve_type(ty);
        debug!(
            "try_index_step(expr={:?}, base_expr={:?}, adjusted_ty={:?}, \
             index_ty={:?})",
            expr, base_expr, adjusted_ty, index_ty
        );

        for unsize in [false, true] {
            let mut self_ty = adjusted_ty;
            if unsize {
                // We only unsize arrays here.
                if let TyKind::Array(element_ty, ct) = adjusted_ty.kind() {
                    let ctx = autoderef.ctx();
                    ctx.table.register_predicate(Obligation::new(
                        ctx.interner(),
                        ObligationCause::new(),
                        ctx.table.param_env,
                        ClauseKind::ConstArgHasType(ct, ctx.types.usize),
                    ));
                    self_ty = Ty::new_slice(ctx.interner(), element_ty);
                } else {
                    continue;
                }
            }

            // If some lookup succeeds, write callee into table and extract index/element
            // type from the method signature.
            // If some lookup succeeded, install method in table
            let input_ty = autoderef.ctx().table.next_ty_var();
            let method =
                autoderef.ctx().try_overloaded_place_op(self_ty, Some(input_ty), PlaceOp::Index);

            if let Some(result) = method {
                debug!("try_index_step: success, using overloaded indexing");
                let method = autoderef.ctx().table.register_infer_ok(result);

                let infer_ok = autoderef.adjust_steps_as_infer_ok();
                let mut adjustments = autoderef.ctx().table.register_infer_ok(infer_ok);
                if let TyKind::Ref(region, _, Mutability::Not) =
                    method.sig.inputs_and_output.inputs()[0].kind()
                {
                    adjustments.push(Adjustment {
                        kind: Adjust::Borrow(AutoBorrow::Ref(AutoBorrowMutability::Not)),
                        target: Ty::new_imm_ref(autoderef.ctx().interner(), region, adjusted_ty),
                    });
                } else {
                    panic!("input to index is not a ref?");
                }
                if unsize {
                    adjustments.push(Adjustment {
                        kind: Adjust::Pointer(PointerCast::Unsize),
                        target: method.sig.inputs_and_output.inputs()[0],
                    });
                }
                autoderef.ctx().write_expr_adj(base_expr, adjustments.into_boxed_slice());

                autoderef.ctx().write_method_resolution(expr, method.def_id, method.args);

                return Some((input_ty, autoderef.ctx().make_overloaded_place_return_type(method)));
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
        base_ty: Ty<'db>,
        opt_rhs_ty: Option<Ty<'db>>,
        op: PlaceOp,
    ) -> Option<InferOk<'db, MethodCallee<'db>>> {
        debug!("try_overloaded_place_op({:?},{:?})", base_ty, op);

        let (Some(imm_tr), imm_op) = (match op {
            PlaceOp::Deref => (self.lang_items.Deref, sym::deref),
            PlaceOp::Index => (self.lang_items.Index, sym::index),
        }) else {
            // Bail if `Deref` or `Index` isn't defined.
            return None;
        };

        // FIXME(trait-system-refactor-initiative#231): we may want to treat
        // opaque types as rigid here to support `impl Deref<Target = impl Index<usize>>`.
        let treat_opaques = TreatNotYetDefinedOpaques::AsInfer;
        self.table.lookup_method_for_operator(
            ObligationCause::new(),
            imm_op,
            imm_tr,
            base_ty,
            opt_rhs_ty,
            treat_opaques,
        )
    }

    pub(super) fn try_mutable_overloaded_place_op(
        table: &InferenceTable<'db>,
        base_ty: Ty<'db>,
        opt_rhs_ty: Option<Ty<'db>>,
        op: PlaceOp,
    ) -> Option<InferOk<'db, MethodCallee<'db>>> {
        debug!("try_mutable_overloaded_place_op({:?},{:?})", base_ty, op);

        let lang_items = table.interner().lang_items();
        let (Some(mut_tr), mut_op) = (match op {
            PlaceOp::Deref => (lang_items.DerefMut, sym::deref_mut),
            PlaceOp::Index => (lang_items.IndexMut, sym::index_mut),
        }) else {
            // Bail if `DerefMut` or `IndexMut` isn't defined.
            return None;
        };

        // We have to replace the operator with the mutable variant for the
        // program to compile, so we don't really have a choice here and want
        // to just try using `DerefMut` even if its not in the item bounds
        // of the opaque.
        let treat_opaques = TreatNotYetDefinedOpaques::AsInfer;
        table.lookup_method_for_operator(
            ObligationCause::new(),
            mut_op,
            mut_tr,
            base_ty,
            opt_rhs_ty,
            treat_opaques,
        )
    }

    pub(super) fn convert_place_op_to_mutable(
        &mut self,
        op: PlaceOp,
        expr: ExprId,
        base_expr: ExprId,
        index_expr: Option<ExprId>,
    ) {
        debug!("convert_place_op_to_mutable({:?}, {:?}, {:?})", op, expr, base_expr);
        if !self.result.method_resolutions.contains_key(&expr) {
            debug!("convert_place_op_to_mutable - builtin, nothing to do");
            return;
        }

        // Need to deref because overloaded place ops take self by-reference.
        let base_ty = self
            .expr_ty_after_adjustments(base_expr)
            .builtin_deref(false)
            .expect("place op takes something that is not a ref");

        let arg_ty = match op {
            PlaceOp::Deref => None,
            PlaceOp::Index => {
                // We would need to recover the `T` used when we resolve `<_ as Index<T>>::index`
                // in try_index_step. This is the arg at index 1.
                //
                // FIXME: rustc does not use the type of `index_expr` with the following explanation.
                //
                // Note: we should *not* use `expr_ty` of index_expr here because autoderef
                // during coercions can cause type of index_expr to differ from `T` (#72002).
                // We also could not use `expr_ty_adjusted` of index_expr because reborrowing
                // during coercions can also cause type of index_expr to differ from `T`,
                // which can potentially cause regionck failure (#74933).
                Some(self.expr_ty_after_adjustments(
                    index_expr.expect("`PlaceOp::Index` should have `index_expr`"),
                ))
            }
        };
        let method = Self::try_mutable_overloaded_place_op(&self.table, base_ty, arg_ty, op);
        let method = match method {
            Some(ok) => self.table.register_infer_ok(ok),
            // Couldn't find the mutable variant of the place op, keep the
            // current, immutable version.
            None => return,
        };
        debug!("convert_place_op_to_mutable: method={:?}", method);
        self.result.method_resolutions.insert(expr, (method.def_id, method.args));

        let TyKind::Ref(region, _, Mutability::Mut) =
            method.sig.inputs_and_output.inputs()[0].kind()
        else {
            panic!("input to mutable place op is not a mut ref?");
        };

        // Convert the autoref in the base expr to mutable with the correct
        // region and mutability.
        let base_expr_ty = self.expr_ty(base_expr);
        let interner = self.interner();
        if let Some(adjustments) = self.result.expr_adjustments.get_mut(&base_expr) {
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
                    adjustment.kind = Adjust::Borrow(AutoBorrow::Ref(mutbl));
                    adjustment.target = Ty::new_ref(interner, region, source, mutbl.into());
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
                *target = method.sig.inputs_and_output.inputs()[0];
            }
        }
    }
}
