use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{
    SpanRangeExt, expr_block, snippet, snippet_block_with_context, snippet_with_applicability, snippet_with_context,
};
use clippy_utils::ty::implements_trait;
use clippy_utils::{
    is_lint_allowed, is_unit_expr, peel_blocks, peel_hir_pat_refs, peel_middle_ty_refs, peel_n_hir_expr_refs,
};
use core::ops::ControlFlow;
use rustc_arena::DroplessArena;
use rustc_errors::{Applicability, Diag};
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::{Visitor, walk_pat};
use rustc_hir::{Arm, Expr, ExprKind, HirId, Node, Pat, PatExpr, PatExprKind, PatKind, QPath, StmtKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, AdtDef, TyCtxt, TypeckResults, VariantDef};
use rustc_span::{Span, sym};

use super::{MATCH_BOOL, SINGLE_MATCH, SINGLE_MATCH_ELSE};

/// Checks if there are comments contained within a span.
/// This is a very "naive" check, as it just looks for the literal characters // and /* in the
/// source text. This won't be accurate if there are potentially expressions contained within the
/// span, e.g. a string literal `"//"`, but we know that this isn't the case for empty
/// match arms.
fn empty_arm_has_comment(cx: &LateContext<'_>, span: Span) -> bool {
    if let Some(ff) = span.get_source_range(cx)
        && let Some(text) = ff.as_str()
    {
        text.as_bytes().windows(2).any(|w| w == b"//" || w == b"/*")
    } else {
        false
    }
}

#[rustfmt::skip]
pub(crate) fn check<'tcx>(cx: &LateContext<'tcx>, ex: &'tcx Expr<'_>, arms: &'tcx [Arm<'_>], expr: &'tcx Expr<'_>, contains_comments: bool) {
    if let [arm1, arm2] = arms
        && !arms.iter().any(|arm| arm.guard.is_some() || arm.pat.span.from_expansion())
        && !expr.span.from_expansion()
        // don't lint for or patterns for now, this makes
        // the lint noisy in unnecessary situations
        && !matches!(arm1.pat.kind, PatKind::Or(..))
    {
        let els = if is_unit_expr(peel_blocks(arm2.body)) && !empty_arm_has_comment(cx, arm2.body.span) {
            None
        } else if let ExprKind::Block(block, _) = arm2.body.kind {
            if matches!((block.stmts, block.expr), ([], Some(_)) | ([_], None)) {
                // single statement/expr "else" block, don't lint
                return;
            }
            // block with 2+ statements or 1 expr and 1+ statement
            Some(arm2.body)
        } else {
            // not a block or an empty block w/ comments, don't lint
            return;
        };

        let typeck = cx.typeck_results();
        if *typeck.expr_ty(ex).peel_refs().kind() != ty::Bool || is_lint_allowed(cx, MATCH_BOOL, ex.hir_id) {
            let mut v = PatVisitor {
                typeck,
                has_enum: false,
            };
            if v.visit_pat(arm2.pat).is_break() {
                return;
            }
            if v.has_enum {
                let cx = PatCtxt {
                    tcx: cx.tcx,
                    typeck,
                    arena: DroplessArena::default(),
                };
                let mut state = PatState::Other;
                if !(state.add_pat(&cx, arm2.pat) || state.add_pat(&cx, arm1.pat)) {
                    // Don't lint if the pattern contains an enum which doesn't have a wild match.
                    return;
                }
            }

            report_single_pattern(cx, ex, arm1, expr, els, contains_comments);
        }
    }
}

fn report_single_pattern(
    cx: &LateContext<'_>,
    ex: &Expr<'_>,
    arm: &Arm<'_>,
    expr: &Expr<'_>,
    els: Option<&Expr<'_>>,
    contains_comments: bool,
) {
    let lint = if els.is_some() { SINGLE_MATCH_ELSE } else { SINGLE_MATCH };
    let ctxt = expr.span.ctxt();
    let note = |diag: &mut Diag<'_, ()>| {
        if contains_comments {
            diag.note("you might want to preserve the comments from inside the `match`");
        }
    };
    let mut app = if contains_comments {
        Applicability::MaybeIncorrect
    } else {
        Applicability::MachineApplicable
    };
    let els_str = els.map_or(String::new(), |els| {
        format!(" else {}", expr_block(cx, els, ctxt, "..", Some(expr.span), &mut app))
    });

    if ex.span.eq_ctxt(expr.span) && snippet(cx, ex.span, "..") == snippet(cx, arm.pat.span, "..") {
        let msg = "this pattern is irrefutable, `match` is useless";
        let (sugg, help) = if is_unit_expr(arm.body) {
            (String::new(), "`match` expression can be removed")
        } else {
            let mut sugg = snippet_block_with_context(cx, arm.body.span, ctxt, "..", Some(expr.span), &mut app).0;
            if let Node::Stmt(stmt) = cx.tcx.parent_hir_node(expr.hir_id)
                && let StmtKind::Expr(_) = stmt.kind
                && match arm.body.kind {
                    ExprKind::Block(block, _) => block.span.from_expansion(),
                    _ => true,
                }
            {
                sugg.push(';');
            }
            (sugg, "try")
        };
        span_lint_and_then(cx, lint, expr.span, msg, |diag| {
            diag.span_suggestion(expr.span, help, sugg, app);
            note(diag);
        });
        return;
    }

    let (pat, pat_ref_count) = peel_hir_pat_refs(arm.pat);
    let (msg, sugg) = if let PatKind::Expr(_) = pat.kind
        && let (ty, ty_ref_count) = peel_middle_ty_refs(cx.typeck_results().expr_ty(ex))
        && let Some(spe_trait_id) = cx.tcx.lang_items().structural_peq_trait()
        && let Some(pe_trait_id) = cx.tcx.lang_items().eq_trait()
        && (ty.is_integral()
            || ty.is_char()
            || ty.is_str()
            || (implements_trait(cx, ty, spe_trait_id, &[]) && implements_trait(cx, ty, pe_trait_id, &[ty.into()])))
    {
        // scrutinee derives PartialEq and the pattern is a constant.
        let pat_ref_count = match pat.kind {
            // string literals are already a reference.
            PatKind::Expr(PatExpr {
                kind: PatExprKind::Lit { lit, negated: false },
                ..
            }) if lit.node.is_str() || lit.node.is_bytestr() => pat_ref_count + 1,
            _ => pat_ref_count,
        };

        // References are implicitly removed when `deref_patterns` are used.
        // They are implicitly added when match ergonomics are used.
        let (ex, ref_or_deref_adjust) = if ty_ref_count > pat_ref_count {
            let ref_count_diff = ty_ref_count - pat_ref_count;

            // Try to remove address of expressions first.
            let (ex, removed) = peel_n_hir_expr_refs(ex, ref_count_diff);

            (ex, String::from(if ref_count_diff == removed { "" } else { "&" }))
        } else {
            (ex, "*".repeat(pat_ref_count - ty_ref_count))
        };

        let msg = "you seem to be trying to use `match` for an equality check. Consider using `if`";
        let sugg = format!(
            "if {} == {}{} {}{els_str}",
            snippet_with_context(cx, ex.span, ctxt, "..", &mut app).0,
            // PartialEq for different reference counts may not exist.
            ref_or_deref_adjust,
            snippet_with_applicability(cx, arm.pat.span, "..", &mut app),
            expr_block(cx, arm.body, ctxt, "..", Some(expr.span), &mut app),
        );
        (msg, sugg)
    } else {
        let msg = "you seem to be trying to use `match` for destructuring a single pattern. Consider using `if let`";
        let sugg = format!(
            "if let {} = {} {}{els_str}",
            snippet_with_applicability(cx, arm.pat.span, "..", &mut app),
            snippet_with_context(cx, ex.span, ctxt, "..", &mut app).0,
            expr_block(cx, arm.body, ctxt, "..", Some(expr.span), &mut app),
        );
        (msg, sugg)
    };

    span_lint_and_then(cx, lint, expr.span, msg, |diag| {
        diag.span_suggestion(expr.span, "try", sugg, app);
        note(diag);
    });
}

struct PatVisitor<'tcx> {
    typeck: &'tcx TypeckResults<'tcx>,
    has_enum: bool,
}
impl<'tcx> Visitor<'tcx> for PatVisitor<'tcx> {
    type Result = ControlFlow<()>;
    fn visit_pat(&mut self, pat: &'tcx Pat<'_>) -> Self::Result {
        if matches!(pat.kind, PatKind::Binding(..)) {
            ControlFlow::Break(())
        } else {
            self.has_enum |= self.typeck.pat_ty(pat).ty_adt_def().is_some_and(AdtDef::is_enum);
            walk_pat(self, pat)
        }
    }
}

/// The context needed to manipulate a `PatState`.
struct PatCtxt<'tcx> {
    tcx: TyCtxt<'tcx>,
    typeck: &'tcx TypeckResults<'tcx>,
    arena: DroplessArena,
}

/// State for tracking whether a match can become non-exhaustive by adding a variant to a contained
/// enum.
///
/// This treats certain std enums as if they will never be extended.
enum PatState<'a> {
    /// Either a wild match or an uninteresting type. Uninteresting types include:
    /// * builtin types (e.g. `i32` or `!`)
    /// * A struct/tuple/array containing only uninteresting types.
    /// * A std enum containing only uninteresting types.
    Wild,
    /// A std enum we know won't be extended. Tracks the states of each variant separately.
    ///
    /// This is not used for `Option` since it uses the current pattern to track it's state.
    StdEnum(&'a mut [PatState<'a>]),
    /// Either the initial state for a pattern or a non-std enum. There is currently no need to
    /// distinguish these cases.
    ///
    /// For non-std enums there's no need to track the state of sub-patterns as the state of just
    /// this pattern on it's own is enough for linting. Consider two cases:
    /// * This enum has no wild match. This case alone is enough to determine we can lint.
    /// * This enum has a wild match and therefore all sub-patterns also have a wild match.
    ///
    /// In both cases the sub patterns are not needed to determine whether to lint.
    Other,
}
impl<'a> PatState<'a> {
    /// Adds a set of patterns as a product type to the current state. Returns whether or not the
    /// current state is a wild match after the merge.
    fn add_product_pat<'tcx>(
        &mut self,
        cx: &'a PatCtxt<'tcx>,
        pats: impl IntoIterator<Item = &'tcx Pat<'tcx>>,
    ) -> bool {
        // Ideally this would actually keep track of the state separately for each pattern. Doing so would
        // require implementing something similar to exhaustiveness checking which is a significant increase
        // in complexity.
        //
        // For now treat this as a wild match only if all the sub-patterns are wild
        let is_wild = pats.into_iter().all(|p| {
            let mut state = Self::Other;
            state.add_pat(cx, p)
        });
        if is_wild {
            *self = Self::Wild;
        }
        is_wild
    }

    /// Attempts to get the state for the enum variant, initializing the current state if necessary.
    fn get_std_enum_variant<'tcx>(
        &mut self,
        cx: &'a PatCtxt<'tcx>,
        adt: AdtDef<'tcx>,
        path: &'tcx QPath<'_>,
        hir_id: HirId,
    ) -> Option<(&mut Self, &'tcx VariantDef)> {
        let states = match self {
            Self::Wild => return None,
            Self::Other => {
                *self = Self::StdEnum(
                    cx.arena
                        .alloc_from_iter(std::iter::repeat_with(|| Self::Other).take(adt.variants().len())),
                );
                let Self::StdEnum(x) = self else {
                    unreachable!();
                };
                x
            },
            Self::StdEnum(x) => x,
        };
        let i = match cx.typeck.qpath_res(path, hir_id) {
            Res::Def(DefKind::Ctor(..), id) => adt.variant_index_with_ctor_id(id),
            Res::Def(DefKind::Variant, id) => adt.variant_index_with_id(id),
            _ => return None,
        };
        Some((&mut states[i.as_usize()], adt.variant(i)))
    }

    fn check_all_wild_enum(&mut self) -> bool {
        if let Self::StdEnum(states) = self
            && states.iter().all(|s| matches!(s, Self::Wild))
        {
            *self = Self::Wild;
            true
        } else {
            false
        }
    }

    #[expect(clippy::similar_names)]
    fn add_struct_pats<'tcx>(
        &mut self,
        cx: &'a PatCtxt<'tcx>,
        pat: &'tcx Pat<'tcx>,
        path: &'tcx QPath<'tcx>,
        single_pat: Option<&'tcx Pat<'tcx>>,
        pats: impl IntoIterator<Item = &'tcx Pat<'tcx>>,
    ) -> bool {
        let ty::Adt(adt, _) = *cx.typeck.pat_ty(pat).kind() else {
            // Should never happen
            *self = Self::Wild;
            return true;
        };
        if adt.is_struct() {
            return if let Some(pat) = single_pat
                && adt.non_enum_variant().fields.len() == 1
            {
                self.add_pat(cx, pat)
            } else {
                self.add_product_pat(cx, pats)
            };
        }
        match cx.tcx.get_diagnostic_name(adt.did()) {
            Some(sym::Option) => {
                if let Some(pat) = single_pat {
                    self.add_pat(cx, pat)
                } else {
                    *self = Self::Wild;
                    true
                }
            },
            Some(sym::Result | sym::Cow) => {
                let Some((state, variant)) = self.get_std_enum_variant(cx, adt, path, pat.hir_id) else {
                    return matches!(self, Self::Wild);
                };
                let is_wild = if let Some(pat) = single_pat
                    && variant.fields.len() == 1
                {
                    state.add_pat(cx, pat)
                } else {
                    state.add_product_pat(cx, pats)
                };
                is_wild && self.check_all_wild_enum()
            },
            _ => matches!(self, Self::Wild),
        }
    }

    /// Adds the pattern into the current state. Returns whether or not the current state is a wild
    /// match after the merge.
    #[expect(clippy::similar_names)]
    fn add_pat<'tcx>(&mut self, cx: &'a PatCtxt<'tcx>, pat: &'tcx Pat<'_>) -> bool {
        match pat.kind {
            PatKind::Expr(PatExpr {
                kind: PatExprKind::Path(_),
                ..
            }) if match *cx.typeck.pat_ty(pat).peel_refs().kind() {
                ty::Adt(adt, _) => adt.is_enum() || (adt.is_struct() && !adt.non_enum_variant().fields.is_empty()),
                ty::Tuple(tys) => !tys.is_empty(),
                ty::Array(_, len) => len.try_to_target_usize(cx.tcx) != Some(1),
                ty::Slice(..) => true,
                _ => false,
            } =>
            {
                matches!(self, Self::Wild)
            },

            PatKind::Guard(..) => {
                matches!(self, Self::Wild)
            },

            // Patterns for things which can only contain a single sub-pattern.
            PatKind::Binding(_, _, _, Some(pat)) | PatKind::Ref(pat, _) | PatKind::Box(pat) | PatKind::Deref(pat) => {
                self.add_pat(cx, pat)
            },
            PatKind::Tuple([sub_pat], pos)
                if pos.as_opt_usize().is_none() || cx.typeck.pat_ty(pat).tuple_fields().len() == 1 =>
            {
                self.add_pat(cx, sub_pat)
            },
            PatKind::Slice([sub_pat], _, []) | PatKind::Slice([], _, [sub_pat])
                if let ty::Array(_, len) = *cx.typeck.pat_ty(pat).kind()
                    && len.try_to_target_usize(cx.tcx) == Some(1) =>
            {
                self.add_pat(cx, sub_pat)
            },

            PatKind::Or(pats) => pats.iter().any(|p| self.add_pat(cx, p)),
            PatKind::Tuple(pats, _) => self.add_product_pat(cx, pats),
            PatKind::Slice(head, _, tail) => self.add_product_pat(cx, head.iter().chain(tail)),

            PatKind::TupleStruct(ref path, pats, _) => self.add_struct_pats(
                cx,
                pat,
                path,
                if let [pat] = pats { Some(pat) } else { None },
                pats.iter(),
            ),
            PatKind::Struct(ref path, pats, _) => self.add_struct_pats(
                cx,
                pat,
                path,
                if let [pat] = pats { Some(pat.pat) } else { None },
                pats.iter().map(|p| p.pat),
            ),

            PatKind::Missing => unreachable!(),
            PatKind::Wild
            | PatKind::Binding(_, _, _, None)
            | PatKind::Expr(_)
            | PatKind::Range(..)
            | PatKind::Never
            | PatKind::Err(_) => {
                *self = PatState::Wild;
                true
            },
        }
    }
}
