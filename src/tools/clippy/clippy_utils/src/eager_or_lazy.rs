//! Utilities for evaluating whether eagerly evaluated expressions can be made lazy and vice versa.
//!
//! Things to consider:
//!  - does the expression have side-effects?
//!  - is the expression computationally expensive?
//!
//! See lints:
//!  - unnecessary-lazy-evaluations
//!  - or-fun-call
//!  - option-if-let-else

use crate::consts::{ConstEvalCtxt, FullInt};
use crate::sym;
use crate::ty::{all_predicates_of, is_copy};
use crate::visitors::is_const_evaluatable;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{Visitor, walk_expr};
use rustc_hir::{BinOpKind, Block, Expr, ExprKind, QPath, UnOp};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_middle::ty::adjustment::Adjust;
use rustc_span::Symbol;
use std::{cmp, ops};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum EagernessSuggestion {
    // The expression is cheap and should be evaluated eagerly
    Eager,
    // The expression may be cheap, so don't suggested lazy evaluation; or the expression may not be safe to switch to
    // eager evaluation.
    NoChange,
    // The expression is likely expensive and should be evaluated lazily.
    Lazy,
    // The expression cannot be placed into a closure.
    ForceNoChange,
}
impl ops::BitOr for EagernessSuggestion {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        cmp::max(self, rhs)
    }
}
impl ops::BitOrAssign for EagernessSuggestion {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

/// Determine the eagerness of the given function call.
fn fn_eagerness(cx: &LateContext<'_>, fn_id: DefId, name: Symbol, have_one_arg: bool) -> EagernessSuggestion {
    use EagernessSuggestion::{Eager, Lazy, NoChange};

    let ty = match cx.tcx.impl_of_method(fn_id) {
        Some(id) => cx.tcx.type_of(id).instantiate_identity(),
        None => return Lazy,
    };

    if (matches!(name, sym::is_empty | sym::len) || name.as_str().starts_with("as_")) && have_one_arg {
        if matches!(
            cx.tcx.crate_name(fn_id.krate),
            sym::std | sym::core | sym::alloc | sym::proc_macro
        ) {
            Eager
        } else {
            NoChange
        }
    } else if let ty::Adt(def, subs) = ty.kind() {
        // Types where the only fields are generic types (or references to) with no trait bounds other
        // than marker traits.
        // Due to the limited operations on these types functions should be fairly cheap.
        if def.variants().iter().flat_map(|v| v.fields.iter()).any(|x| {
            matches!(
                cx.tcx.type_of(x.did).instantiate_identity().peel_refs().kind(),
                ty::Param(_)
            )
        }) && all_predicates_of(cx.tcx, fn_id).all(|(pred, _)| match pred.kind().skip_binder() {
            ty::ClauseKind::Trait(pred) => cx.tcx.trait_def(pred.trait_ref.def_id).is_marker,
            _ => true,
        }) && subs.types().all(|x| matches!(x.peel_refs().kind(), ty::Param(_)))
        {
            // Limit the function to either `(self) -> bool` or `(&self) -> bool`
            match &**cx
                .tcx
                .fn_sig(fn_id)
                .instantiate_identity()
                .skip_binder()
                .inputs_and_output
            {
                [arg, res] if !arg.is_mutable_ptr() && arg.peel_refs() == ty && res.is_bool() => NoChange,
                _ => Lazy,
            }
        } else {
            Lazy
        }
    } else {
        Lazy
    }
}

fn res_has_significant_drop(res: Res, cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    if let Res::Def(DefKind::Ctor(..) | DefKind::Variant | DefKind::Enum | DefKind::Struct, _)
    | Res::SelfCtor(_)
    | Res::SelfTyAlias { .. } = res
    {
        cx.typeck_results()
            .expr_ty(e)
            .has_significant_drop(cx.tcx, cx.typing_env())
    } else {
        false
    }
}

#[expect(clippy::too_many_lines)]
fn expr_eagerness<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) -> EagernessSuggestion {
    struct V<'cx, 'tcx> {
        cx: &'cx LateContext<'tcx>,
        eagerness: EagernessSuggestion,
    }

    impl<'tcx> Visitor<'tcx> for V<'_, 'tcx> {
        fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
            use EagernessSuggestion::{ForceNoChange, Lazy, NoChange};
            if self.eagerness == ForceNoChange {
                return;
            }

            // Autoderef through a user-defined `Deref` impl can have side-effects,
            // so don't suggest changing it.
            if self
                .cx
                .typeck_results()
                .expr_adjustments(e)
                .iter()
                .any(|adj| matches!(adj.kind, Adjust::Deref(Some(_))))
            {
                self.eagerness |= NoChange;
                return;
            }

            match e.kind {
                ExprKind::Call(
                    &Expr {
                        kind: ExprKind::Path(ref path),
                        hir_id,
                        ..
                    },
                    args,
                ) => match self.cx.qpath_res(path, hir_id) {
                    res @ (Res::Def(DefKind::Ctor(..) | DefKind::Variant, _) | Res::SelfCtor(_)) => {
                        if res_has_significant_drop(res, self.cx, e) {
                            self.eagerness = ForceNoChange;
                            return;
                        }
                    },
                    Res::Def(_, id) if self.cx.tcx.is_promotable_const_fn(id) => (),
                    // No need to walk the arguments here, `is_const_evaluatable` already did
                    Res::Def(..) if is_const_evaluatable(self.cx, e) => {
                        self.eagerness |= NoChange;
                        return;
                    },
                    Res::Def(_, id) => match path {
                        QPath::Resolved(_, p) => {
                            self.eagerness |=
                                fn_eagerness(self.cx, id, p.segments.last().unwrap().ident.name, !args.is_empty());
                        },
                        QPath::TypeRelative(_, name) => {
                            self.eagerness |= fn_eagerness(self.cx, id, name.ident.name, !args.is_empty());
                        },
                        QPath::LangItem(..) => self.eagerness = Lazy,
                    },
                    _ => self.eagerness = Lazy,
                },
                // No need to walk the arguments here, `is_const_evaluatable` already did
                ExprKind::MethodCall(..) if is_const_evaluatable(self.cx, e) => {
                    self.eagerness |= NoChange;
                    return;
                },
                #[expect(clippy::match_same_arms)] // arm pattern can't be merged due to `ref`, see rust#105778
                ExprKind::Struct(path, ..) => {
                    if res_has_significant_drop(self.cx.qpath_res(path, e.hir_id), self.cx, e) {
                        self.eagerness = ForceNoChange;
                        return;
                    }
                },
                ExprKind::Path(ref path) => {
                    if res_has_significant_drop(self.cx.qpath_res(path, e.hir_id), self.cx, e) {
                        self.eagerness = ForceNoChange;
                        return;
                    }
                },
                ExprKind::MethodCall(name, ..) => {
                    self.eagerness |= self
                        .cx
                        .typeck_results()
                        .type_dependent_def_id(e.hir_id)
                        .map_or(Lazy, |id| fn_eagerness(self.cx, id, name.ident.name, true));
                },
                ExprKind::Index(_, e, _) => {
                    let ty = self.cx.typeck_results().expr_ty_adjusted(e);
                    if is_copy(self.cx, ty) && !ty.is_ref() {
                        self.eagerness |= NoChange;
                    } else {
                        self.eagerness = Lazy;
                    }
                },

                // `-i32::MIN` panics with overflow checks
                ExprKind::Unary(UnOp::Neg, right) if ConstEvalCtxt::new(self.cx).eval(right).is_none() => {
                    self.eagerness |= NoChange;
                },

                // Custom `Deref` impl might have side effects
                ExprKind::Unary(UnOp::Deref, e)
                    if self.cx.typeck_results().expr_ty(e).builtin_deref(true).is_none() =>
                {
                    self.eagerness |= NoChange;
                },
                // Dereferences should be cheap, but dereferencing a raw pointer earlier may not be safe.
                ExprKind::Unary(UnOp::Deref, e) if !self.cx.typeck_results().expr_ty(e).is_raw_ptr() => (),
                ExprKind::Unary(UnOp::Deref, _) => self.eagerness |= NoChange,
                ExprKind::Unary(_, e)
                    if matches!(
                        self.cx.typeck_results().expr_ty(e).kind(),
                        ty::Bool | ty::Int(_) | ty::Uint(_),
                    ) => {},

                // `>>` and `<<` panic when the right-hand side is greater than or equal to the number of bits in the
                // type of the left-hand side, or is negative.
                // We intentionally only check if the right-hand isn't a constant, because even if the suggestion would
                // overflow with constants, the compiler emits an error for it and the programmer will have to fix it.
                // Thus, we would realistically only delay the lint.
                ExprKind::Binary(op, _, right)
                    if matches!(op.node, BinOpKind::Shl | BinOpKind::Shr)
                        && ConstEvalCtxt::new(self.cx).eval(right).is_none() =>
                {
                    self.eagerness |= NoChange;
                },

                ExprKind::Binary(op, left, right)
                    if matches!(op.node, BinOpKind::Div | BinOpKind::Rem)
                        && let right_ty = self.cx.typeck_results().expr_ty(right)
                        && let ecx = ConstEvalCtxt::new(self.cx)
                        && let left = ecx.eval(left)
                        && let right = ecx.eval(right).and_then(|c| c.int_value(self.cx.tcx, right_ty))
                        && matches!(
                            (left, right),
                            // `1 / x`: x might be zero
                            (_, None)
                            // `x / -1`: x might be T::MIN
                            | (None, Some(FullInt::S(-1)))
                        ) =>
                {
                    self.eagerness |= NoChange;
                },

                // Similar to `>>` and `<<`, we only want to avoid linting entirely if either side is unknown and the
                // compiler can't emit an error for an overflowing expression.
                // Suggesting eagerness for `true.then(|| i32::MAX + 1)` is okay because the compiler will emit an
                // error and it's good to have the eagerness warning up front when the user fixes the logic error.
                ExprKind::Binary(op, left, right)
                    if matches!(op.node, BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul)
                        && !self.cx.typeck_results().expr_ty(e).is_floating_point()
                        && let ecx = ConstEvalCtxt::new(self.cx)
                        && (ecx.eval(left).is_none() || ecx.eval(right).is_none()) =>
                {
                    self.eagerness |= NoChange;
                },

                ExprKind::Binary(_, lhs, rhs)
                    if self.cx.typeck_results().expr_ty(lhs).is_primitive()
                        && self.cx.typeck_results().expr_ty(rhs).is_primitive() => {},

                // Can't be moved into a closure
                ExprKind::Break(..)
                | ExprKind::Continue(_)
                | ExprKind::Ret(_)
                | ExprKind::Become(_)
                | ExprKind::InlineAsm(_)
                | ExprKind::Yield(..)
                | ExprKind::Err(_) => {
                    self.eagerness = ForceNoChange;
                    return;
                },

                // Memory allocation, custom operator, loop, or call to an unknown function
                ExprKind::Unary(..) | ExprKind::Binary(..) | ExprKind::Loop(..) | ExprKind::Call(..) => {
                    self.eagerness = Lazy;
                },

                ExprKind::ConstBlock(_)
                | ExprKind::Array(_)
                | ExprKind::Tup(_)
                | ExprKind::Use(..)
                | ExprKind::Lit(_)
                | ExprKind::Cast(..)
                | ExprKind::Type(..)
                | ExprKind::DropTemps(_)
                | ExprKind::Let(..)
                | ExprKind::If(..)
                | ExprKind::Match(..)
                | ExprKind::Closure { .. }
                | ExprKind::Field(..)
                | ExprKind::AddrOf(..)
                | ExprKind::Repeat(..)
                | ExprKind::Block(Block { stmts: [], .. }, _)
                | ExprKind::OffsetOf(..)
                | ExprKind::UnsafeBinderCast(..) => (),

                // Assignment might be to a local defined earlier, so don't eagerly evaluate.
                // Blocks with multiple statements might be expensive, so don't eagerly evaluate.
                // TODO: Actually check if either of these are true here.
                ExprKind::Assign(..) | ExprKind::AssignOp(..) | ExprKind::Block(..) => self.eagerness |= NoChange,
            }
            walk_expr(self, e);
        }
    }

    let mut v = V {
        cx,
        eagerness: EagernessSuggestion::Eager,
    };
    v.visit_expr(e);
    v.eagerness
}

/// Whether the given expression should be changed to evaluate eagerly
pub fn switch_to_eager_eval<'tcx>(cx: &'_ LateContext<'tcx>, expr: &'tcx Expr<'_>) -> bool {
    expr_eagerness(cx, expr) == EagernessSuggestion::Eager
}

/// Whether the given expression should be changed to evaluate lazily
pub fn switch_to_lazy_eval<'tcx>(cx: &'_ LateContext<'tcx>, expr: &'tcx Expr<'_>) -> bool {
    expr_eagerness(cx, expr) == EagernessSuggestion::Lazy
}
