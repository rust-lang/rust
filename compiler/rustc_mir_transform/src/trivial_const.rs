use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{Block, BlockCheckMode, ExprKind, Lit, StmtKind, UnOp, UnsafeSource};
use rustc_middle::mir::{Const, ConstValue};
use rustc_middle::ty::consts::lit_to_mir_constant;
use rustc_middle::ty::{AnonConstKind, LitToConstInput, Ty, TyCtxt, TypeVisitableExt};

/// If the given def is a trivial const, returns the value and type the const evaluates to.
///
/// A "trivial const" is a const which can be easily proven to evaluate successfully, and the value
/// that it evaluates to can be easily found without going through the usual MIR phases for a const.
///
/// Currently, we support two forms of trivial const.
///
/// The base case is this:
/// ```
/// const A: usize = 0;
/// ```
/// In addition to assignment of literals, assignments of trivial consts are also considered
/// trivial consts. In this case, both `A` and `B` are trivial:
/// ```
/// const A: usize = 0;
/// const B: usize = A;
/// ```
/// Additionally:
/// * Control flow cannot panic, we don't have any calls or assertions.
/// * The value of the const is already computed, so it cannot fail.
///
/// We identify triviality at the HIR level. We allow the following shapes: literals, negated
/// literals, coercion casts, const paths, item-only blocks, and tail-less unit blocks. Why these
/// shapes? Because the analysis used to be done on MIR, and in the base case looked for
/// this general pattern:
/// ```text
/// const A: usize = {
///     let mut _0: usize;
///
///     bb0: {
///         _0 = const 0_usize;
///         return;
///     }
/// }
/// ```
/// and those HIR shapes all boil down to something like that at the MIR level.
///
/// This analysis must be conservative.
/// * False negatives are fine: ordinary path, MIR is built and analyzed as normal. The cost is
///   increased compile time, and possibly minor changes in `-Zdump-mir` output and cycle errors.
/// * False positives are bad: they cause all sorts of missed checks (borrowck, unsafety, etc.).
///   Also, `eval_to_const_value_raw` uses the returned `ConstValue` for various things (e.g. array
///   lengths).
///
pub(crate) fn trivial_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: LocalDefId,
) -> Option<(ConstValue, Ty<'tcx>)> {
    enum Inner {
        Literal(Lit, bool),
        NoTailExpr,
    }

    match tcx.def_kind(def) {
        DefKind::AssocConst | DefKind::Const => (),
        DefKind::AnonConst if tcx.anon_const_kind(def) != AnonConstKind::NonTypeSystemInline => (),
        _ => return None,
    }

    // If there are impossible clauses then MIR passes will replace the body with
    // `unreachable` causing const eval errors when trying to evaluate the body. For
    // now we avoid using trivial consts for such bodies so that the behaviour doesn't
    // change.
    if crate::impossible_clauses::has_impossible_clauses(tcx, def.into()) {
        return None;
    }

    if !tcx.opaque_types_defined_by(def).is_empty() {
        return None;
    }

    // Bail if there is no body. That happens for direct consts.
    let Some(body) = tcx.hir_maybe_body_owned_by(def) else {
        assert!(tcx.features().gca_min_const_items());
        return None;
    };

    // Bail if there are arg patterns.
    if !body.params.is_empty() {
        return None;
    }

    // Bail if there are any type errors, to let MIR build produce an error body.
    let typeck_results = tcx.typeck(def);
    if typeck_results.tainted_by_errors.is_some() {
        return None;
    }

    let mut expr = body.value;
    let leaf = loop {
        // Don't allow adjustments, e.g. coercions, derefs.
        if !typeck_results.expr_adjustments(expr).is_empty() {
            return None;
        }

        match expr.kind {
            // Literals are fine.
            ExprKind::Lit(lit) => break Inner::Literal(lit, false),

            // Non-overloaded unary negation is fine.
            ExprKind::Unary(UnOp::Neg, inner) => match inner.kind {
                ExprKind::Lit(lit) if !typeck_results.is_method_call(expr) => {
                    break Inner::Literal(lit, true);
                }
                _ => return None,
            },

            // Coercion casts without user-provided types are fine because they don't leave any
            // trace behind in MIR.
            ExprKind::Cast(source_expr, cast_ty) => {
                if !typeck_results.is_coercion_cast(source_expr.hir_id) {
                    return None;
                }
                if typeck_results.user_provided_types().get(cast_ty.hir_id).is_some() {
                    return None;
                }
                expr = source_expr;
            }

            // Trivial, non-generic named constants are fine.
            ExprKind::Path(qpath) => {
                let res = typeck_results.qpath_res(&qpath, expr.hir_id);
                let Res::Def(DefKind::Const | DefKind::AssocConst, def_id) = res else {
                    return None;
                };
                if !typeck_results.node_args(expr.hir_id).is_empty() {
                    return None;
                }
                return tcx.trivial_const(def_id);
            }

            // Most blocks are fine because they add scopes but no statements or basic blocks.
            ExprKind::Block(block, _label) => {
                // Deconstruct `block` so that if extra fields are added in the future we must
                // consider them here.
                let &Block {
                    stmts,
                    expr: block_expr,
                    hir_id: _,
                    rules,
                    span: _,
                    targeted_by_break,
                } = block;

                // Bail if any `break` is involved.
                if targeted_by_break {
                    return None;
                }
                // Bail on `unsafe { ... }`, because `check_unsafety` is skipped for trivial
                // consts.
                if matches!(rules, BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided)) {
                    return None;
                }
                for stmt in stmts {
                    // Allow items here because THIR discards them. Any such items are separate HIR
                    // owners with their own bodies that are checked separately even if this const
                    // is skipped. Any kind of item can occur here, most commonly in macro output
                    // such as `const _: () = { <item> ... };`.
                    let StmtKind::Item(..) = stmt.kind else {
                        return None;
                    };
                }
                if let Some(block_expr) = block_expr {
                    // Unwrap the block.
                    expr = block_expr;
                } else {
                    break Inner::NoTailExpr;
                }
            }
            _ => return None,
        }
    };

    let ty = typeck_results.expr_ty(expr);

    // Bail if opaque types are present.
    if ty.has_opaque_types() || tcx.type_of(def).instantiate_identity().has_opaque_types() {
        return None;
    }

    match leaf {
        Inner::Literal(lit, neg) => {
            // Check the constant is trivial.
            let input = LitToConstInput { lit: lit.node, ty: Some(ty), neg };
            let const_ = lit_to_mir_constant(tcx, input);
            match const_ {
                Const::Ty(..) => None,
                Const::Val(v, ty) => Some((v, ty)),
                Const::Unevaluated(..) => unreachable!(),
            }
        }
        Inner::NoTailExpr => {
            if ty.is_unit() {
                // No tail expression and a unit destination: builder emits `_0 = const ()`.
                Some((ConstValue::ZeroSized, ty))
            } else {
                None
            }
        }
    }
}
