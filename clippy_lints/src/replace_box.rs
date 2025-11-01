use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::{MaybeDef, MaybeResPath};
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::implements_trait;
use clippy_utils::{is_default_equivalent_call, local_is_initialized};
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::smallvec::SmallVec;
use rustc_errors::Applicability;
use rustc_hir::{Body, BodyId, Expr, ExprKind, HirId, LangItem, QPath};
use rustc_hir_typeck::expr_use_visitor::{Delegate, ExprUseVisitor, PlaceBase, PlaceWithHirId};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::place::ProjectionKind;
use rustc_middle::mir::FakeReadCause;
use rustc_middle::ty;
use rustc_session::impl_lint_pass;
use rustc_span::{Symbol, sym};

declare_clippy_lint! {
    /// ### What it does
    /// Detects assignments of `Default::default()` or `Box::new(value)`
    /// to a place of type `Box<T>`.
    ///
    /// ### Why is this bad?
    /// This incurs an extra heap allocation compared to assigning the boxed
    /// storage.
    ///
    /// ### Example
    /// ```no_run
    /// let mut b = Box::new(1u32);
    /// b = Default::default();
    /// ```
    /// Use instead:
    /// ```no_run
    /// let mut b = Box::new(1u32);
    /// *b = Default::default();
    /// ```
    #[clippy::version = "1.92.0"]
    pub REPLACE_BOX,
    perf,
    "assigning a newly created box to `Box<T>` is inefficient"
}

#[derive(Default)]
pub struct ReplaceBox {
    consumed_locals: FxHashSet<HirId>,
    loaded_bodies: SmallVec<[BodyId; 2]>,
}

impl ReplaceBox {
    fn get_consumed_locals(&mut self, cx: &LateContext<'_>) -> &FxHashSet<HirId> {
        if let Some(body_id) = cx.enclosing_body
            && !self.loaded_bodies.contains(&body_id)
        {
            self.loaded_bodies.push(body_id);
            ExprUseVisitor::for_clippy(
                cx,
                cx.tcx.hir_body_owner_def_id(body_id),
                MovedVariablesCtxt {
                    consumed_locals: &mut self.consumed_locals,
                },
            )
            .consume_body(cx.tcx.hir_body(body_id))
            .into_ok();
        }

        &self.consumed_locals
    }
}

impl_lint_pass!(ReplaceBox => [REPLACE_BOX]);

impl LateLintPass<'_> for ReplaceBox {
    fn check_body_post(&mut self, _: &LateContext<'_>, body: &Body<'_>) {
        if self.loaded_bodies.first().is_some_and(|&x| x == body.id()) {
            self.consumed_locals.clear();
            self.loaded_bodies.clear();
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &'_ Expr<'_>) {
        if let ExprKind::Assign(lhs, rhs, _) = &expr.kind
            && !lhs.span.from_expansion()
            && !rhs.span.from_expansion()
            && let lhs_ty = cx.typeck_results().expr_ty(lhs)
            && let Some(inner_ty) = lhs_ty.boxed_ty()
            // No diagnostic for late-initialized locals
            && lhs.res_local_id().is_none_or(|local| local_is_initialized(cx, local))
            // No diagnostic if this is a local that has been moved, or the field
            // of a local that has been moved, or several chained field accesses of a local
            && local_base(lhs).is_none_or(|(base_id, _)| {
                !self.get_consumed_locals(cx).contains(&base_id)
            })
        {
            if let Some(default_trait_id) = cx.tcx.get_diagnostic_item(sym::Default)
                && implements_trait(cx, inner_ty, default_trait_id, &[])
                && is_default_call(cx, rhs)
            {
                span_lint_and_then(
                    cx,
                    REPLACE_BOX,
                    expr.span,
                    "creating a new box with default content",
                    |diag| {
                        let mut app = Applicability::MachineApplicable;
                        let suggestion = format!(
                            "{} = Default::default()",
                            Sugg::hir_with_applicability(cx, lhs, "_", &mut app).deref()
                        );

                        diag.note("this creates a needless allocation").span_suggestion(
                            expr.span,
                            "replace existing content with default instead",
                            suggestion,
                            app,
                        );
                    },
                );
            }

            if inner_ty.is_sized(cx.tcx, cx.typing_env())
                && let Some(rhs_inner) = get_box_new_payload(cx, rhs)
            {
                span_lint_and_then(cx, REPLACE_BOX, expr.span, "creating a new box", |diag| {
                    let mut app = Applicability::MachineApplicable;
                    let suggestion = format!(
                        "{} = {}",
                        Sugg::hir_with_applicability(cx, lhs, "_", &mut app).deref(),
                        Sugg::hir_with_context(cx, rhs_inner, expr.span.ctxt(), "_", &mut app),
                    );

                    diag.note("this creates a needless allocation").span_suggestion(
                        expr.span,
                        "replace existing content with inner value instead",
                        suggestion,
                        app,
                    );
                });
            }
        }
    }
}

fn is_default_call(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    matches!(expr.kind, ExprKind::Call(func, _args) if is_default_equivalent_call(cx, func, Some(expr)))
}

fn get_box_new_payload<'tcx>(cx: &LateContext<'_>, expr: &Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    if let ExprKind::Call(box_new, [arg]) = expr.kind
        && let ExprKind::Path(QPath::TypeRelative(ty, seg)) = box_new.kind
        && seg.ident.name == sym::new
        && ty.basic_res().is_lang_item(cx, LangItem::OwnedBox)
    {
        Some(arg)
    } else {
        None
    }
}

struct MovedVariablesCtxt<'a> {
    consumed_locals: &'a mut FxHashSet<HirId>,
}

impl<'tcx> Delegate<'tcx> for MovedVariablesCtxt<'_> {
    fn consume(&mut self, cmt: &PlaceWithHirId<'tcx>, _: HirId) {
        if let PlaceBase::Local(id) = cmt.place.base
            && let mut projections = cmt
                .place
                .projections
                .iter()
                .filter(|x| matches!(x.kind, ProjectionKind::Deref))
            // Either no deref or multiple derefs
            && (projections.next().is_none() || projections.next().is_some())
        {
            self.consumed_locals.insert(id);
        }
    }

    fn use_cloned(&mut self, _: &PlaceWithHirId<'tcx>, _: HirId) {}

    fn borrow(&mut self, _: &PlaceWithHirId<'tcx>, _: HirId, _: ty::BorrowKind) {}

    fn mutate(&mut self, _: &PlaceWithHirId<'tcx>, _: HirId) {}

    fn fake_read(&mut self, _: &PlaceWithHirId<'tcx>, _: FakeReadCause, _: HirId) {}
}

/// A local place followed by optional fields
type IdFields = (HirId, Vec<Symbol>);

/// If `expr` is a local variable with optional field accesses, return it.
fn local_base(expr: &Expr<'_>) -> Option<IdFields> {
    match expr.kind {
        ExprKind::Path(qpath) => qpath.res_local_id().map(|id| (id, Vec::new())),
        ExprKind::Field(expr, field) => local_base(expr).map(|(id, mut fields)| {
            fields.push(field.name);
            (id, fields)
        }),
        _ => None,
    }
}
