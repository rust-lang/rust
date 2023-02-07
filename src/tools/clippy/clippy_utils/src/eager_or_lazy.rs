//! Utilities for evaluating whether eagerly evaluated expressions can be made lazy and vice versa.
//!
//! Things to consider:
//!  - has the expression side-effects?
//!  - is the expression computationally expensive?
//!
//! See lints:
//!  - unnecessary-lazy-evaluations
//!  - or-fun-call
//!  - option-if-let-else

use crate::ty::{all_predicates_of, is_copy};
use crate::visitors::is_const_evaluatable;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::{walk_expr, Visitor};
use rustc_hir::{def_id::DefId, Block, Expr, ExprKind, QPath, UnOp};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, PredicateKind};
use rustc_span::{sym, Symbol};
use std::cmp;
use std::ops;

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
    let name = name.as_str();

    let ty = match cx.tcx.impl_of_method(fn_id) {
        Some(id) => cx.tcx.type_of(id).subst_identity(),
        None => return Lazy,
    };

    if (name.starts_with("as_") || name == "len" || name == "is_empty") && have_one_arg {
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
        if def
            .variants()
            .iter()
            .flat_map(|v| v.fields.iter())
            .any(|x| matches!(cx.tcx.type_of(x.did).subst_identity().peel_refs().kind(), ty::Param(_)))
            && all_predicates_of(cx.tcx, fn_id).all(|(pred, _)| match pred.kind().skip_binder() {
                PredicateKind::Clause(ty::Clause::Trait(pred)) => cx.tcx.trait_def(pred.trait_ref.def_id).is_marker,
                _ => true,
            })
            && subs.types().all(|x| matches!(x.peel_refs().kind(), ty::Param(_)))
        {
            // Limit the function to either `(self) -> bool` or `(&self) -> bool`
            match &**cx.tcx.fn_sig(fn_id).subst_identity().skip_binder().inputs_and_output {
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
    if let Res::Def(DefKind::Ctor(..) | DefKind::Variant, _) | Res::SelfCtor(_) = res {
        cx.typeck_results()
            .expr_ty(e)
            .has_significant_drop(cx.tcx, cx.param_env)
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

    impl<'cx, 'tcx> Visitor<'tcx> for V<'cx, 'tcx> {
        fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
            use EagernessSuggestion::{ForceNoChange, Lazy, NoChange};
            if self.eagerness == ForceNoChange {
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
                ExprKind::Index(_, e) => {
                    let ty = self.cx.typeck_results().expr_ty_adjusted(e);
                    if is_copy(self.cx, ty) && !ty.is_ref() {
                        self.eagerness |= NoChange;
                    } else {
                        self.eagerness = Lazy;
                    }
                },

                // Dereferences should be cheap, but dereferencing a raw pointer earlier may not be safe.
                ExprKind::Unary(UnOp::Deref, e) if !self.cx.typeck_results().expr_ty(e).is_unsafe_ptr() => (),
                ExprKind::Unary(UnOp::Deref, _) => self.eagerness |= NoChange,

                ExprKind::Unary(_, e)
                    if matches!(
                        self.cx.typeck_results().expr_ty(e).kind(),
                        ty::Bool | ty::Int(_) | ty::Uint(_),
                    ) => {},
                ExprKind::Binary(_, lhs, rhs)
                    if self.cx.typeck_results().expr_ty(lhs).is_primitive()
                        && self.cx.typeck_results().expr_ty(rhs).is_primitive() => {},

                // Can't be moved into a closure
                ExprKind::Break(..)
                | ExprKind::Continue(_)
                | ExprKind::Ret(_)
                | ExprKind::InlineAsm(_)
                | ExprKind::Yield(..)
                | ExprKind::Err => {
                    self.eagerness = ForceNoChange;
                    return;
                },

                // Memory allocation, custom operator, loop, or call to an unknown function
                ExprKind::Box(_)
                | ExprKind::Unary(..)
                | ExprKind::Binary(..)
                | ExprKind::Loop(..)
                | ExprKind::Call(..) => self.eagerness = Lazy,

                ExprKind::ConstBlock(_)
                | ExprKind::Array(_)
                | ExprKind::Tup(_)
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
                | ExprKind::Struct(..)
                | ExprKind::Repeat(..)
                | ExprKind::Block(Block { stmts: [], .. }, _) => (),

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
