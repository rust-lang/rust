use super::FnCtxt;
use crate::expr_use_visitor::{ConsumeMode, Delegate, ExprUseVisitor};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::hir_id::HirId;
use rustc_hir::intravisit::{self, Visitor};
use rustc_middle::hir::place::{PlaceBase, PlaceWithHirId};
use rustc_middle::ty;
use rustc_span::hygiene::SyntaxContext;
use rustc_span::symbol::Ident;
use rustc_span::Span;

// Checks for type ascriptions in lvalue contexts, i.e. in situations such as
// (x : T) = ..., &(x : T), &mut (x : T) and (x : T).foo(...), where
// type ascriptions can potentially lead to type unsoundness problems
struct TypeAscriptionValidator<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    finder: TypeAscriptionFinder<'a>,
}

impl<'a, 'tcx> Delegate<'tcx> for TypeAscriptionValidator<'a, 'tcx> {
    fn consume(
        &mut self,
        _place_with_id: &PlaceWithHirId<'tcx>,
        _diag_expr_id: hir::HirId,
        _mode: ConsumeMode,
    ) {
    }

    fn borrow(
        &mut self,
        place_with_id: &PlaceWithHirId<'tcx>,
        _diag_expr_id: hir::HirId,
        bk: ty::BorrowKind,
    ) {
        debug!(
            "TypeAscriptionkind::borrow(place: {:?}, borrow_kind: {:?})",
            place_with_id.place, bk
        );

        if let PlaceBase::Rvalue(id) = place_with_id.place.base {
            if let Some(expr) = self.fcx.tcx.hir().maybe_expr(id) {
                debug!("expr behind place: {:?}", expr);
                self.finder.visit_expr(expr);
                if let Some(ascr_expr) = self.finder.found_type_ascr() {
                    if let hir::ExprKind::Type(ref e, ref t) = ascr_expr.kind {
                        let span = ascr_expr.span;
                        let mut err = self.fcx.tcx.sess.struct_span_err(
                            span,
                            "type ascriptions are not allowed in lvalue contexts",
                        );

                        if let Some((span, ident)) = self.maybe_get_span_ident_for_diagnostics(e) {
                            if let Ok(ty_str) =
                                self.fcx.tcx.sess.source_map().span_to_snippet(t.span)
                            {
                                err.span_suggestion(
                                    span,
                                    "try to use the type ascription when creating the local variable",
                                    format!("let {}: {}", ident, ty_str),
                                    Applicability::MaybeIncorrect,
                                );
                            }
                        }
                        err.emit();
                    }
                }
                self.finder.reset();
            }
        }
    }

    fn mutate(&mut self, assignee_place: &PlaceWithHirId<'tcx>, diag_expr_id: hir::HirId) {
        debug!(
            "TypeAscription::mutate(assignee_place: {:?}, diag_expr_id: {:?})",
            assignee_place, diag_expr_id
        );

        if let PlaceBase::Rvalue(id) = assignee_place.place.base {
            if let Some(expr) = self.fcx.tcx.hir().maybe_expr(id) {
                debug!("expr behind place: {:?}", expr);
                if let hir::ExprKind::Type(_, _) = expr.kind {
                    let span = expr.span;
                    let mut err = self.fcx.tcx.sess.struct_span_err(
                        span,
                        "type ascriptions are not allowed in lvalue contexts",
                    );
                    err.emit();
                }
            }
        }
    }
}

impl TypeAscriptionValidator<'a, 'tcx> {
    // Try to get the necessary information for an error suggestion, in case the
    // place on which the type ascription was used is a local variable.
    fn maybe_get_span_ident_for_diagnostics(&self, e: &hir::Expr<'_>) -> Option<(Span, Ident)> {
        match e.kind {
            hir::ExprKind::Path(hir::QPath::Resolved(_, path)) => {
                let hir::Path { res, segments, .. } = path;
                if let Res::Local(id) = res {
                    // Span for the definition of the local variable
                    let span = self.fcx.tcx.hir().span(*id);
                    let source_map = self.fcx.tcx.sess.source_map();

                    if let Ok(file_lines) = source_map.span_to_lines(span) {
                        let source_file = file_lines.file;
                        let lines = file_lines.lines;

                        // Only create suggestion if the assignment operator is on the first line
                        let line_bounds_range =
                            lines.get(0).and_then(|l| Some(source_file.line_bounds(l.line_index)));
                        if let Some(range) = line_bounds_range {
                            let line_span =
                                Span::new(range.start, range.end, SyntaxContext::root());
                            if let Ok(line_string) = source_map.span_to_snippet(line_span) {
                                if line_string.contains("=") {
                                    let span_til_eq = source_map.span_until_char(line_span, '=');
                                    if segments.len() == 1 {
                                        let ident = segments[0].ident;
                                        return Some((span_til_eq, ident));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        None
    }
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub fn analyze_type_ascriptions(&self, body: &'tcx hir::Body<'tcx>) {
        let body_owner_def_id = self.tcx.hir().body_owner_def_id(body.id());
        let finder = TypeAscriptionFinder::new();
        let mut delegate = TypeAscriptionValidator { fcx: self, finder };
        ExprUseVisitor::new(
            &mut delegate,
            &self.infcx,
            body_owner_def_id,
            self.param_env,
            &self.typeck_results.borrow(),
        )
        .consume_body(body);
    }
}

struct TypeAscriptionFinder<'tcx> {
    found: Option<&'tcx hir::Expr<'tcx>>,
    found_ids: Vec<HirId>,
}

impl<'tcx> TypeAscriptionFinder<'tcx> {
    fn new() -> TypeAscriptionFinder<'tcx> {
        TypeAscriptionFinder { found: None, found_ids: vec![] }
    }

    fn found_type_ascr(&self) -> Option<&'tcx hir::Expr<'tcx>> {
        self.found
    }

    fn reset(&mut self) {
        self.found = None;
    }
}

impl Visitor<'tcx> for TypeAscriptionFinder<'tcx> {
    type Map = intravisit::ErasedMap<'tcx>;

    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        intravisit::NestedVisitorMap::None
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        match expr.kind {
            hir::ExprKind::Type(..) => {
                if !self.found_ids.contains(&expr.hir_id) {
                    self.found = Some(expr);
                    self.found_ids.push(expr.hir_id);
                    return;
                }
            }
            _ => intravisit::walk_expr(self, expr),
        }
    }
}
