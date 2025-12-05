//! Inference of binary and unary operators.

use std::collections::hash_map;

use hir_def::{GenericParamId, TraitId, hir::ExprId};
use intern::{Symbol, sym};
use rustc_ast_ir::Mutability;
use rustc_type_ir::inherent::{IntoKind, Ty as _};
use syntax::ast::{ArithOp, BinaryOp, UnaryOp};
use tracing::debug;

use crate::{
    Adjust, Adjustment, AutoBorrow,
    infer::{AllowTwoPhase, AutoBorrowMutability, Expectation, InferenceContext, expr::ExprIsRead},
    method_resolution::{MethodCallee, TreatNotYetDefinedOpaques},
    next_solver::{
        GenericArgs, TraitRef, Ty, TyKind,
        fulfill::NextSolverError,
        infer::traits::{Obligation, ObligationCause},
        obligation_ctxt::ObligationCtxt,
    },
};

impl<'a, 'db> InferenceContext<'a, 'db> {
    /// Checks a `a <op>= b`
    pub(crate) fn infer_assign_op_expr(
        &mut self,
        expr: ExprId,
        op: ArithOp,
        lhs: ExprId,
        rhs: ExprId,
    ) -> Ty<'db> {
        let (lhs_ty, rhs_ty, return_ty) =
            self.infer_overloaded_binop(expr, lhs, rhs, BinaryOp::Assignment { op: Some(op) });

        let category = BinOpCategory::from(op);
        let ty = if !lhs_ty.is_ty_var()
            && !rhs_ty.is_ty_var()
            && is_builtin_binop(lhs_ty, rhs_ty, category)
        {
            self.enforce_builtin_binop_types(lhs_ty, rhs_ty, category);
            self.types.unit
        } else {
            return_ty
        };

        self.check_lhs_assignable(lhs);

        ty
    }

    /// Checks a potentially overloaded binary operator.
    pub(crate) fn infer_binop_expr(
        &mut self,
        expr: ExprId,
        op: BinaryOp,
        lhs_expr: ExprId,
        rhs_expr: ExprId,
    ) -> Ty<'db> {
        debug!(
            "check_binop(expr.hir_id={:?}, expr={:?}, op={:?}, lhs_expr={:?}, rhs_expr={:?})",
            expr, expr, op, lhs_expr, rhs_expr
        );

        match op {
            BinaryOp::LogicOp(_) => {
                // && and || are a simple case.
                self.infer_expr_coerce(
                    lhs_expr,
                    &Expectation::HasType(self.types.bool),
                    ExprIsRead::Yes,
                );
                let lhs_diverges = self.diverges;
                self.infer_expr_coerce(
                    rhs_expr,
                    &Expectation::HasType(self.types.bool),
                    ExprIsRead::Yes,
                );

                // Depending on the LHS' value, the RHS can never execute.
                self.diverges = lhs_diverges;

                self.types.bool
            }
            _ => {
                // Otherwise, we always treat operators as if they are
                // overloaded. This is the way to be most flexible w/r/t
                // types that get inferred.
                let (lhs_ty, rhs_ty, return_ty) =
                    self.infer_overloaded_binop(expr, lhs_expr, rhs_expr, op);

                // Supply type inference hints if relevant. Probably these
                // hints should be enforced during select as part of the
                // `consider_unification_despite_ambiguity` routine, but this
                // more convenient for now.
                //
                // The basic idea is to help type inference by taking
                // advantage of things we know about how the impls for
                // scalar types are arranged. This is important in a
                // scenario like `1_u32 << 2`, because it lets us quickly
                // deduce that the result type should be `u32`, even
                // though we don't know yet what type 2 has and hence
                // can't pin this down to a specific impl.
                let category = BinOpCategory::from(op);
                if !lhs_ty.is_ty_var()
                    && !rhs_ty.is_ty_var()
                    && is_builtin_binop(lhs_ty, rhs_ty, category)
                {
                    let builtin_return_ty =
                        self.enforce_builtin_binop_types(lhs_ty, rhs_ty, category);
                    _ = self.demand_eqtype(expr.into(), builtin_return_ty, return_ty);
                    builtin_return_ty
                } else {
                    return_ty
                }
            }
        }
    }

    fn enforce_builtin_binop_types(
        &mut self,
        lhs_ty: Ty<'db>,
        rhs_ty: Ty<'db>,
        category: BinOpCategory,
    ) -> Ty<'db> {
        debug_assert!(is_builtin_binop(lhs_ty, rhs_ty, category));

        // Special-case a single layer of referencing, so that things like `5.0 + &6.0f32` work.
        // (See https://github.com/rust-lang/rust/issues/57447.)
        let (lhs_ty, rhs_ty) = (deref_ty_if_possible(lhs_ty), deref_ty_if_possible(rhs_ty));

        match category {
            BinOpCategory::Shortcircuit => {
                self.demand_suptype(self.types.bool, lhs_ty);
                self.demand_suptype(self.types.bool, rhs_ty);
                self.types.bool
            }

            BinOpCategory::Shift => {
                // result type is same as LHS always
                lhs_ty
            }

            BinOpCategory::Math | BinOpCategory::Bitwise => {
                // both LHS and RHS and result will have the same type
                self.demand_suptype(lhs_ty, rhs_ty);
                lhs_ty
            }

            BinOpCategory::Comparison => {
                // both LHS and RHS and result will have the same type
                self.demand_suptype(lhs_ty, rhs_ty);
                self.types.bool
            }
        }
    }

    fn infer_overloaded_binop(
        &mut self,
        expr: ExprId,
        lhs_expr: ExprId,
        rhs_expr: ExprId,
        op: BinaryOp,
    ) -> (Ty<'db>, Ty<'db>, Ty<'db>) {
        debug!("infer_overloaded_binop(expr.hir_id={:?}, op={:?})", expr, op);

        let lhs_ty = match op {
            BinaryOp::Assignment { .. } => {
                // rust-lang/rust#52126: We have to use strict
                // equivalence on the LHS of an assign-op like `+=`;
                // overwritten or mutably-borrowed places cannot be
                // coerced to a supertype.
                self.infer_expr_no_expect(lhs_expr, ExprIsRead::Yes)
            }
            _ => {
                // Find a suitable supertype of the LHS expression's type, by coercing to
                // a type variable, to pass as the `Self` to the trait, avoiding invariant
                // trait matching creating lifetime constraints that are too strict.
                // e.g., adding `&'a T` and `&'b T`, given `&'x T: Add<&'x T>`, will result
                // in `&'a T <: &'x T` and `&'b T <: &'x T`, instead of `'a = 'b = 'x`.
                let lhs_ty = self.infer_expr_no_expect(lhs_expr, ExprIsRead::No);
                let fresh_var = self.table.next_ty_var();
                self.demand_coerce(lhs_expr, lhs_ty, fresh_var, AllowTwoPhase::No, ExprIsRead::No)
            }
        };
        let lhs_ty = self.table.resolve_vars_with_obligations(lhs_ty);

        // N.B., as we have not yet type-checked the RHS, we don't have the
        // type at hand. Make a variable to represent it. The whole reason
        // for this indirection is so that, below, we can check the expr
        // using this variable as the expected type, which sometimes lets
        // us do better coercions than we would be able to do otherwise,
        // particularly for things like `String + &String`.
        let rhs_ty_var = self.table.next_ty_var();
        let result = self.lookup_op_method(
            lhs_ty,
            Some((rhs_expr, rhs_ty_var)),
            self.lang_item_for_bin_op(op),
        );

        // see `NB` above
        let rhs_ty =
            self.infer_expr_coerce(rhs_expr, &Expectation::HasType(rhs_ty_var), ExprIsRead::No);
        let rhs_ty = self.table.resolve_vars_with_obligations(rhs_ty);

        let return_ty = match result {
            Ok(method) => {
                let by_ref_binop = !is_op_by_value(op);
                if (matches!(op, BinaryOp::Assignment { .. }) || by_ref_binop)
                    && let TyKind::Ref(_, _, mutbl) =
                        method.sig.inputs_and_output.inputs()[0].kind()
                {
                    let mutbl = AutoBorrowMutability::new(mutbl, AllowTwoPhase::Yes);
                    let autoref = Adjustment {
                        kind: Adjust::Borrow(AutoBorrow::Ref(mutbl)),
                        target: method.sig.inputs_and_output.inputs()[0],
                    };
                    self.write_expr_adj(lhs_expr, Box::new([autoref]));
                }
                if by_ref_binop
                    && let TyKind::Ref(_, _, mutbl) =
                        method.sig.inputs_and_output.inputs()[1].kind()
                {
                    // Allow two-phase borrows for binops in initial deployment
                    // since they desugar to methods
                    let mutbl = AutoBorrowMutability::new(mutbl, AllowTwoPhase::Yes);

                    let autoref = Adjustment {
                        kind: Adjust::Borrow(AutoBorrow::Ref(mutbl)),
                        target: method.sig.inputs_and_output.inputs()[1],
                    };
                    // HACK(eddyb) Bypass checks due to reborrows being in
                    // some cases applied on the RHS, on top of which we need
                    // to autoref, which is not allowed by write_expr_adj.
                    // self.write_expr_adj(rhs_expr, Box::new([autoref]));
                    match self.result.expr_adjustments.entry(rhs_expr) {
                        hash_map::Entry::Occupied(mut entry) => {
                            let mut adjustments = Vec::from(std::mem::take(entry.get_mut()));
                            adjustments.reserve_exact(1);
                            adjustments.push(autoref);
                            entry.insert(adjustments.into_boxed_slice());
                        }
                        hash_map::Entry::Vacant(entry) => {
                            entry.insert(Box::new([autoref]));
                        }
                    };
                }
                self.write_method_resolution(expr, method.def_id, method.args);

                method.sig.output()
            }
            Err(_errors) => {
                // FIXME: Report diagnostic.
                self.types.error
            }
        };

        (lhs_ty, rhs_ty, return_ty)
    }

    pub(crate) fn infer_user_unop(
        &mut self,
        ex: ExprId,
        operand_ty: Ty<'db>,
        op: UnaryOp,
    ) -> Ty<'db> {
        match self.lookup_op_method(operand_ty, None, self.lang_item_for_unop(op)) {
            Ok(method) => {
                self.write_method_resolution(ex, method.def_id, method.args);
                method.sig.output()
            }
            Err(_errors) => {
                // FIXME: Report diagnostic.
                self.types.error
            }
        }
    }

    fn lookup_op_method(
        &mut self,
        lhs_ty: Ty<'db>,
        opt_rhs: Option<(ExprId, Ty<'db>)>,
        (opname, trait_did): (Symbol, Option<TraitId>),
    ) -> Result<MethodCallee<'db>, Vec<NextSolverError<'db>>> {
        let Some(trait_did) = trait_did else {
            // Bail if the operator trait is not defined.
            return Err(vec![]);
        };

        debug!(
            "lookup_op_method(lhs_ty={:?}, opname={:?}, trait_did={:?})",
            lhs_ty, opname, trait_did
        );

        let opt_rhs_ty = opt_rhs.map(|it| it.1);
        let cause = ObligationCause::new();

        // We don't consider any other candidates if this lookup fails
        // so we can freely treat opaque types as inference variables here
        // to allow more code to compile.
        let treat_opaques = TreatNotYetDefinedOpaques::AsInfer;
        let method = self.table.lookup_method_for_operator(
            cause.clone(),
            opname,
            trait_did,
            lhs_ty,
            opt_rhs_ty,
            treat_opaques,
        );
        match method {
            Some(ok) => {
                let method = self.table.register_infer_ok(ok);
                self.table.select_obligations_where_possible();
                Ok(method)
            }
            None => {
                // Guide inference for the RHS expression if it's provided --
                // this will allow us to better error reporting, at the expense
                // of making some error messages a bit more specific.
                if let Some((rhs_expr, rhs_ty)) = opt_rhs
                    && rhs_ty.is_ty_var()
                {
                    self.infer_expr_coerce(rhs_expr, &Expectation::HasType(rhs_ty), ExprIsRead::No);
                }

                // Construct an obligation `self_ty : Trait<input_tys>`
                let args = GenericArgs::for_item(
                    self.interner(),
                    trait_did.into(),
                    |param_idx, param_id, _| match param_id {
                        GenericParamId::LifetimeParamId(_) | GenericParamId::ConstParamId(_) => {
                            unreachable!("did not expect operand trait to have lifetime/const args")
                        }
                        GenericParamId::TypeParamId(_) => {
                            if param_idx == 0 {
                                lhs_ty.into()
                            } else {
                                opt_rhs_ty.expect("expected RHS for binop").into()
                            }
                        }
                    },
                );
                let obligation = Obligation::new(
                    self.interner(),
                    cause,
                    self.table.param_env,
                    TraitRef::new_from_args(self.interner(), trait_did.into(), args),
                );
                let mut ocx = ObligationCtxt::new(self.infcx());
                ocx.register_obligation(obligation);
                Err(ocx.evaluate_obligations_error_on_ambiguity())
            }
        }
    }

    fn lang_item_for_bin_op(&self, op: BinaryOp) -> (Symbol, Option<TraitId>) {
        let (method_name, trait_lang_item) =
            crate::lang_items::lang_items_for_bin_op(self.lang_items, op)
                .expect("invalid operator provided");
        (method_name, trait_lang_item)
    }

    fn lang_item_for_unop(&self, op: UnaryOp) -> (Symbol, Option<TraitId>) {
        let (method_name, trait_lang_item) = match op {
            UnaryOp::Not => (sym::not, self.lang_items.Not),
            UnaryOp::Neg => (sym::neg, self.lang_items.Neg),
            UnaryOp::Deref => panic!("Deref is not overloadable"),
        };
        (method_name, trait_lang_item)
    }
}

// Binary operator categories. These categories summarize the behavior
// with respect to the builtin operations supported.
#[derive(Clone, Copy)]
enum BinOpCategory {
    /// &&, || -- cannot be overridden
    Shortcircuit,

    /// <<, >> -- when shifting a single integer, rhs can be any
    /// integer type. For simd, types must match.
    Shift,

    /// +, -, etc -- takes equal types, produces same type as input,
    /// applicable to ints/floats/simd
    Math,

    /// &, |, ^ -- takes equal types, produces same type as input,
    /// applicable to ints/floats/simd/bool
    Bitwise,

    /// ==, !=, etc -- takes equal types, produces bools, except for simd,
    /// which produce the input type
    Comparison,
}

impl From<BinaryOp> for BinOpCategory {
    fn from(op: BinaryOp) -> BinOpCategory {
        match op {
            BinaryOp::LogicOp(_) => BinOpCategory::Shortcircuit,
            BinaryOp::ArithOp(op) | BinaryOp::Assignment { op: Some(op) } => op.into(),
            BinaryOp::CmpOp(_) => BinOpCategory::Comparison,
            BinaryOp::Assignment { op: None } => unreachable!(
                "assignment is lowered into `Expr::Assignment`, not into `Expr::BinaryOp`"
            ),
        }
    }
}

impl From<ArithOp> for BinOpCategory {
    fn from(op: ArithOp) -> BinOpCategory {
        use ArithOp::*;
        match op {
            Shl | Shr => BinOpCategory::Shift,
            Add | Sub | Mul | Div | Rem => BinOpCategory::Math,
            BitXor | BitAnd | BitOr => BinOpCategory::Bitwise,
        }
    }
}

/// Returns `true` if the binary operator takes its arguments by value.
fn is_op_by_value(op: BinaryOp) -> bool {
    !matches!(op, BinaryOp::CmpOp(_))
}

/// Dereferences a single level of immutable referencing.
fn deref_ty_if_possible(ty: Ty<'_>) -> Ty<'_> {
    match ty.kind() {
        TyKind::Ref(_, ty, Mutability::Not) => ty,
        _ => ty,
    }
}

/// Returns `true` if this is a built-in arithmetic operation (e.g.,
/// u32 + u32, i16x4 == i16x4) and false if these types would have to be
/// overloaded to be legal. There are two reasons that we distinguish
/// builtin operations from overloaded ones (vs trying to drive
/// everything uniformly through the trait system and intrinsics or
/// something like that):
///
/// 1. Builtin operations can trivially be evaluated in constants.
/// 2. For comparison operators applied to SIMD types the result is
///    not of type `bool`. For example, `i16x4 == i16x4` yields a
///    type like `i16x4`. This means that the overloaded trait
///    `PartialEq` is not applicable.
///
/// Reason #2 is the killer. I tried for a while to always use
/// overloaded logic and just check the types in constants/codegen after
/// the fact, and it worked fine, except for SIMD types. -nmatsakis
fn is_builtin_binop<'db>(lhs: Ty<'db>, rhs: Ty<'db>, category: BinOpCategory) -> bool {
    // Special-case a single layer of referencing, so that things like `5.0 + &6.0f32` work.
    // (See https://github.com/rust-lang/rust/issues/57447.)
    let (lhs, rhs) = (deref_ty_if_possible(lhs), deref_ty_if_possible(rhs));

    match category {
        BinOpCategory::Shortcircuit => true,
        BinOpCategory::Shift => lhs.is_integral() && rhs.is_integral(),
        BinOpCategory::Math => {
            lhs.is_integral() && rhs.is_integral()
                || lhs.is_floating_point() && rhs.is_floating_point()
        }
        BinOpCategory::Bitwise => {
            lhs.is_integral() && rhs.is_integral()
                || lhs.is_floating_point() && rhs.is_floating_point()
                || lhs.is_bool() && rhs.is_bool()
        }
        BinOpCategory::Comparison => lhs.is_scalar() && rhs.is_scalar(),
    }
}
