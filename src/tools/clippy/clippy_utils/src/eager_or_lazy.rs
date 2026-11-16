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

use crate::consts::ConstEvalCtxt;
use crate::sym;
use crate::ty::all_clauses_of;
use crate::visitors::{Visitable as _, is_const_evaluatable};
use core::ops::ControlFlow::{self, Break, Continue};
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{Visitor, walk_expr};
use rustc_hir::{BinOpKind, Block, Expr, ExprKind, QPath, UnOp};
use rustc_lint::LateContext;
use rustc_middle::ty::adjustment::{Adjust, DerefAdjustKind};
use rustc_middle::ty::{self, TyCtxt};
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
fn fn_eagerness(tcx: TyCtxt<'_>, fn_id: DefId, name: Symbol, have_one_arg: bool) -> EagernessSuggestion {
    use EagernessSuggestion::{Eager, Lazy, NoChange};

    let ty = match tcx.impl_of_assoc(fn_id) {
        Some(id) => tcx.type_of(id).instantiate_identity().skip_norm_wip(),
        None => return Lazy,
    };

    if (matches!(name, sym::is_empty | sym::len) || name.as_str().starts_with("as_")) && have_one_arg {
        if matches!(
            tcx.crate_name(fn_id.krate),
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
                tcx.type_of(x.did)
                    .instantiate_identity()
                    .skip_norm_wip()
                    .peel_refs()
                    .kind(),
                ty::Param(_)
            )
        }) && all_clauses_of(tcx, fn_id).all(|(clause, _)| match clause.kind().skip_binder() {
            ty::ClauseKind::Trait(pred) => tcx.trait_def(pred.trait_ref.def_id).is_marker,
            _ => true,
        }) && subs.types().all(|x| matches!(x.peel_refs().kind(), ty::Param(_)))
        {
            // Limit the function to either `(self) -> bool` or `(&self) -> bool`
            match &**tcx
                .fn_sig(fn_id)
                .instantiate_identity()
                .skip_norm_wip()
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

fn res_has_significant_drop(res: Res, ecx: &ConstEvalCtxt<'_>, e: &Expr<'_>) -> bool {
    if let Res::Def(DefKind::Ctor(..) | DefKind::Variant | DefKind::Enum | DefKind::Struct, _)
    | Res::SelfCtor(_)
    | Res::SelfTyAlias { .. } = res
    {
        ecx.typeck.expr_ty(e).has_significant_drop(ecx.tcx, ecx.typing_env)
    } else {
        false
    }
}

struct ExprEagernessVisitor<'tcx> {
    ecx: ConstEvalCtxt<'tcx>,
    eagerness: EagernessSuggestion,
}
impl<'tcx> ExprEagernessVisitor<'tcx> {
    fn visit_unary(&mut self, op: UnOp, e: &'tcx Expr<'tcx>) -> ControlFlow<()> {
        use EagernessSuggestion::{Lazy, NoChange};

        let ty = self.ecx.typeck.expr_ty(e).kind();
        let (deref_ty, deref) = match ty {
            &ty::Ref(_, ty, _) => {
                // TODO(@Jarcho): Allow eager evaluation starting with 1.102.
                if op != UnOp::Deref {
                    self.eagerness |= NoChange;
                }
                (ty.kind(), true)
            },
            ty => (ty, false),
        };
        match op {
            UnOp::Neg => match *deref_ty {
                ty::Int(_) => {
                    if self.ecx.eval_deref(e, deref).is_some() {
                        // rustc's `arithmetic_overflow` lint will catch this case.
                        return Continue(());
                    }
                    // Possible overflow.
                    self.eagerness |= NoChange;
                },
                ty::Float(_) => {},
                _ => self.eagerness = Lazy,
            },
            UnOp::Deref => match *ty {
                ty::Adt(def, _) if def.is_box() => {},
                ty::Ref(..) => {},
                // Raw pointer dereferences have validity invariants which may not be
                // met if moved earlier. Everything else is a custom deref which should be
                // cheap, but we don't know for sure.
                _ => self.eagerness |= NoChange,
            },
            UnOp::Not => match *deref_ty {
                ty::Int(_) | ty::Uint(_) | ty::Bool => {},
                _ => self.eagerness |= NoChange,
            },
        }
        self.visit_expr(e)
    }

    fn visit_binary(&mut self, op: BinOpKind, lhs: &'tcx Expr<'tcx>, rhs: &'tcx Expr<'tcx>) -> ControlFlow<()> {
        use EagernessSuggestion::{Lazy, NoChange};

        let lhs_ty = self.ecx.typeck.expr_ty(lhs);
        let (lhs_ty, lhs_deref) = match *lhs_ty.kind() {
            ty::Ref(_, lhs_ty, _) => {
                // TODO(@Jarcho): Allow eager evaluation starting with 1.102.
                self.eagerness |= NoChange;
                (lhs_ty, true)
            },
            _ => (lhs_ty, false),
        };
        let rhs_ty = self.ecx.typeck.expr_ty(rhs);
        let (rhs_ty, rhs_deref) = match *rhs_ty.kind() {
            ty::Ref(_, rhs_ty, _) => {
                // TODO(@Jarcho): Allow eager evaluation starting with 1.102.
                self.eagerness |= NoChange;
                (rhs_ty, true)
            },
            _ => (rhs_ty, false),
        };

        if let BinOpKind::Eq | BinOpKind::Ne | BinOpKind::Lt | BinOpKind::Le | BinOpKind::Gt | BinOpKind::Ge = op {
            if lhs_deref == rhs_deref {
                let mut lhs_ty = lhs_ty;
                let mut rhs_ty = rhs_ty;
                while let ty::Ref(_, lhs, _) = *lhs_ty.kind()
                    && let ty::Ref(_, rhs, _) = *rhs_ty.kind()
                {
                    lhs_ty = lhs;
                    rhs_ty = rhs;
                }
                match *lhs_ty.kind() {
                    ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) if lhs_ty == rhs_ty => {},
                    ty::RawPtr(..) if rhs_ty.is_raw_ptr() => {},
                    _ => self.eagerness = Lazy,
                }
            } else {
                self.eagerness = Lazy;
            }
        } else if let Some((is_signed, ty)) = match (lhs_ty.kind(), rhs_ty.kind()) {
            // If both sides are an integer type then we can assume either the types are
            // the same, or we have a shift operator. Since we don't care about the types
            // in that case just take the lhs's type.
            (&ty::Int(ty), ty::Int(_) | ty::Uint(_)) => Some((true, ty)),
            (&ty::Uint(ty), ty::Int(_) | ty::Uint(_)) => Some((false, ty.to_signed())),
            (ty::Float(_), ty::Float(_)) | (ty::Bool, ty::Bool) => None,
            _ => {
                self.eagerness = Lazy;
                None
            },
        } {
            // For operators which can panic we will only suggest eager evaluation if the
            // expression would be caught by rustc's `arithmetic_overflow` or
            // `unconditional_panic` lints.
            match (op, is_signed) {
                (BinOpKind::Div | BinOpKind::Rem, true) if let Some(rhs_val) = self.ecx.eval_deref(rhs, rhs_deref) => {
                    if rhs_val.to_int(self.ecx.tcx, ty) == Some(-1) {
                        if self.ecx.eval_deref(lhs, lhs_deref).is_some() {
                            return Continue(());
                        }
                        // Possible overflow.
                        self.eagerness |= NoChange;
                    }
                    return self.visit_expr(rhs);
                },
                (BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul, _)
                    if self.ecx.eval_deref(rhs, rhs_deref).is_some() =>
                {
                    return if self.ecx.eval_deref(lhs, lhs_deref).is_some() {
                        Continue(())
                    } else {
                        // Possible overflow.
                        self.eagerness |= NoChange;
                        self.visit_expr(lhs)
                    };
                },
                (BinOpKind::Div | BinOpKind::Rem, false) | (BinOpKind::Shl | BinOpKind::Shr, _)
                    if self.ecx.eval_deref(rhs, rhs_deref).is_some() =>
                {
                    return self.visit_expr(lhs);
                },
                (BinOpKind::Or | BinOpKind::And | BinOpKind::BitAnd | BinOpKind::BitOr | BinOpKind::BitXor, _) => {},
                _ => {
                    // Possible division-by-zero, overflow, or over/negative shift.
                    self.eagerness |= NoChange;
                },
            }
        }
        (lhs, rhs).visit(self)
    }
}
impl<'tcx> Visitor<'tcx> for ExprEagernessVisitor<'tcx> {
    type Result = ControlFlow<()>;

    #[expect(clippy::too_many_lines)]
    fn visit_expr(&mut self, e: &'tcx Expr<'_>) -> Self::Result {
        use EagernessSuggestion::{ForceNoChange, Lazy, NoChange};

        // Autoderef through a user-defined `Deref` impl can have side-effects,
        // so don't suggest changing it.
        if self
            .ecx
            .typeck
            .expr_adjustments(e)
            .iter()
            .any(|adj| matches!(adj.kind, Adjust::Deref(DerefAdjustKind::Overloaded(_))))
        {
            self.eagerness |= NoChange;
        }

        match e.kind {
            ExprKind::Call(
                &Expr {
                    kind: ExprKind::Path(ref path),
                    hir_id,
                    ..
                },
                args,
            ) => match self.ecx.typeck.qpath_res(path, hir_id) {
                res @ (Res::Def(DefKind::Ctor(..) | DefKind::Variant, _) | Res::SelfCtor(_)) => {
                    if res_has_significant_drop(res, &self.ecx, e) {
                        self.eagerness = ForceNoChange;
                        return Break(());
                    }
                },
                Res::Def(_, id) if self.ecx.tcx.is_promotable_const_fn(id) => (),
                // No need to walk the arguments here, `is_const_evaluatable` already did
                Res::Def(..) if is_const_evaluatable(self.ecx.tcx, self.ecx.typeck, e) => {
                    self.eagerness |= NoChange;
                    return Break(());
                },
                Res::Def(_, id) => match path {
                    QPath::Resolved(_, p) => {
                        self.eagerness |= fn_eagerness(
                            self.ecx.tcx,
                            id,
                            p.segments.last().unwrap().ident.name,
                            !args.is_empty(),
                        );
                    },
                    QPath::TypeRelative(_, name) => {
                        self.eagerness |= fn_eagerness(self.ecx.tcx, id, name.ident.name, !args.is_empty());
                    },
                },
                _ => self.eagerness = Lazy,
            },
            // No need to walk the arguments here, `is_const_evaluatable` already did
            ExprKind::MethodCall(..) if is_const_evaluatable(self.ecx.tcx, self.ecx.typeck, e) => {
                self.eagerness |= NoChange;
                return Break(());
            },
            #[expect(clippy::match_same_arms)] // arm pattern can't be merged due to `ref`, see rust#105778
            ExprKind::Struct(path, ..) => {
                if res_has_significant_drop(self.ecx.typeck.qpath_res(path, e.hir_id), &self.ecx, e) {
                    self.eagerness = ForceNoChange;
                    return Break(());
                }
            },
            ExprKind::Path(ref path) => {
                if res_has_significant_drop(self.ecx.typeck.qpath_res(path, e.hir_id), &self.ecx, e) {
                    self.eagerness = ForceNoChange;
                    return Break(());
                }
            },
            ExprKind::MethodCall(name, ..) => {
                self.eagerness |= self
                    .ecx
                    .typeck
                    .type_dependent_def_id(e.hir_id)
                    .map_or(Lazy, |id| fn_eagerness(self.ecx.tcx, id, name.ident.name, true));
            },
            ExprKind::Index(_, e, _) => {
                let ty = self.ecx.typeck.expr_ty_adjusted(e);
                if self.ecx.tcx.type_is_copy_modulo_regions(self.ecx.typing_env, ty) && !ty.is_ref() {
                    self.eagerness |= NoChange;
                } else {
                    self.eagerness = Lazy;
                }
            },

            ExprKind::Unary(op, e) => return self.visit_unary(op, e),
            ExprKind::Binary(op, lhs, rhs) => return self.visit_binary(op.node, lhs, rhs),

            // Can't be moved into a closure
            ExprKind::Break(..)
            | ExprKind::Continue(_)
            | ExprKind::Ret(_)
            | ExprKind::Become(_)
            | ExprKind::InlineAsm(_)
            | ExprKind::Yield(..)
            | ExprKind::Err(_) => {
                self.eagerness = ForceNoChange;
                return Break(());
            },

            ExprKind::Loop(..) | ExprKind::Call(..) => {
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
        walk_expr(self, e)
    }
}

fn expr_eagerness<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) -> EagernessSuggestion {
    let mut v = ExprEagernessVisitor {
        ecx: ConstEvalCtxt::new(cx),
        eagerness: EagernessSuggestion::Eager,
    };
    let _ = v.visit_expr(e);
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
