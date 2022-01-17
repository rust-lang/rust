use Context::*;

use rustc_errors::{struct_span_err, Applicability};
use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{Destination, Movability, Node};
use rustc_middle::hir::map::Map;
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_span::hygiene::DesugaringKind;
use rustc_span::Span;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Context {
    Normal,
    Loop(hir::LoopSource),
    Closure(Span),
    AsyncClosure(Span),
    LabeledBlock,
    AnonConst,
}

#[derive(Copy, Clone)]
struct CheckLoopVisitor<'a, 'hir> {
    sess: &'a Session,
    hir_map: Map<'hir>,
    cx: Context,
}

fn check_mod_loops(tcx: TyCtxt<'_>, module_def_id: LocalDefId) {
    tcx.hir().visit_item_likes_in_module(
        module_def_id,
        &mut CheckLoopVisitor { sess: &tcx.sess, hir_map: tcx.hir(), cx: Normal }.as_deep_visitor(),
    );
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { check_mod_loops, ..*providers };
}

impl<'a, 'hir> Visitor<'hir> for CheckLoopVisitor<'a, 'hir> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.hir_map
    }

    fn visit_anon_const(&mut self, c: &'hir hir::AnonConst) {
        self.with_context(AnonConst, |v| intravisit::walk_anon_const(v, c));
    }

    fn visit_expr(&mut self, e: &'hir hir::Expr<'hir>) {
        match e.kind {
            hir::ExprKind::Loop(ref b, _, source, _) => {
                self.with_context(Loop(source), |v| v.visit_block(&b));
            }
            hir::ExprKind::Closure(_, ref function_decl, b, span, movability) => {
                let cx = if let Some(Movability::Static) = movability {
                    AsyncClosure(span)
                } else {
                    Closure(span)
                };
                self.visit_fn_decl(&function_decl);
                self.with_context(cx, |v| v.visit_nested_body(b));
            }
            hir::ExprKind::Block(ref b, Some(_label)) => {
                self.with_context(LabeledBlock, |v| v.visit_block(&b));
            }
            hir::ExprKind::Break(break_label, ref opt_expr) => {
                if let Some(e) = opt_expr {
                    self.visit_expr(e);
                }

                if self.require_label_in_labeled_block(e.span, &break_label, "break") {
                    // If we emitted an error about an unlabeled break in a labeled
                    // block, we don't need any further checking for this break any more
                    return;
                }

                let loop_id = match break_label.target_id {
                    Ok(loop_id) => Some(loop_id),
                    Err(hir::LoopIdError::OutsideLoopScope) => None,
                    Err(hir::LoopIdError::UnlabeledCfInWhileCondition) => {
                        self.emit_unlabled_cf_in_while_condition(e.span, "break");
                        None
                    }
                    Err(hir::LoopIdError::UnresolvedLabel) => None,
                };

                if let Some(Node::Block(_)) = loop_id.and_then(|id| self.hir_map.find(id)) {
                    return;
                }

                if let Some(break_expr) = opt_expr {
                    let (head, loop_label, loop_kind) = if let Some(loop_id) = loop_id {
                        match self.hir_map.expect_expr(loop_id).kind {
                            hir::ExprKind::Loop(_, label, source, sp) => {
                                (Some(sp), label, Some(source))
                            }
                            ref r => {
                                span_bug!(e.span, "break label resolved to a non-loop: {:?}", r)
                            }
                        }
                    } else {
                        (None, None, None)
                    };
                    match loop_kind {
                        None | Some(hir::LoopSource::Loop) => (),
                        Some(kind) => {
                            let mut err = struct_span_err!(
                                self.sess,
                                e.span,
                                E0571,
                                "`break` with value from a `{}` loop",
                                kind.name()
                            );
                            err.span_label(
                                e.span,
                                "can only break with a value inside `loop` or breakable block",
                            );
                            if let Some(head) = head {
                                err.span_label(
                                    head,
                                    &format!(
                                        "you can't `break` with a value in a `{}` loop",
                                        kind.name()
                                    ),
                                );
                            }
                            err.span_suggestion(
                                e.span,
                                &format!(
                                    "use `break` on its own without a value inside this `{}` loop",
                                    kind.name(),
                                ),
                                format!(
                                    "break{}",
                                    break_label
                                        .label
                                        .map_or_else(String::new, |l| format!(" {}", l.ident))
                                ),
                                Applicability::MaybeIncorrect,
                            );
                            if let (Some(label), None) = (loop_label, break_label.label) {
                                match break_expr.kind {
                                    hir::ExprKind::Path(hir::QPath::Resolved(
                                        None,
                                        hir::Path {
                                            segments: [segment],
                                            res: hir::def::Res::Err,
                                            ..
                                        },
                                    )) if label.ident.to_string()
                                        == format!("'{}", segment.ident) =>
                                    {
                                        // This error is redundant, we will have already emitted a
                                        // suggestion to use the label when `segment` wasn't found
                                        // (hence the `Res::Err` check).
                                        err.delay_as_bug();
                                    }
                                    _ => {
                                        err.span_suggestion(
                                            break_expr.span,
                                            "alternatively, you might have meant to use the \
                                             available loop label",
                                            label.ident.to_string(),
                                            Applicability::MaybeIncorrect,
                                        );
                                    }
                                }
                            }
                            err.emit();
                        }
                    }
                }

                self.require_break_cx("break", e.span);
            }
            hir::ExprKind::Continue(destination) => {
                self.require_label_in_labeled_block(e.span, &destination, "continue");

                match destination.target_id {
                    Ok(loop_id) => {
                        if let Node::Block(block) = self.hir_map.find(loop_id).unwrap() {
                            struct_span_err!(
                                self.sess,
                                e.span,
                                E0696,
                                "`continue` pointing to a labeled block"
                            )
                            .span_label(e.span, "labeled blocks cannot be `continue`'d")
                            .span_label(block.span, "labeled block the `continue` points to")
                            .emit();
                        }
                    }
                    Err(hir::LoopIdError::UnlabeledCfInWhileCondition) => {
                        self.emit_unlabled_cf_in_while_condition(e.span, "continue");
                    }
                    Err(_) => {}
                }
                self.require_break_cx("continue", e.span)
            }
            _ => intravisit::walk_expr(self, e),
        }
    }
}

impl<'a, 'hir> CheckLoopVisitor<'a, 'hir> {
    fn with_context<F>(&mut self, cx: Context, f: F)
    where
        F: FnOnce(&mut CheckLoopVisitor<'a, 'hir>),
    {
        let old_cx = self.cx;
        self.cx = cx;
        f(self);
        self.cx = old_cx;
    }

    fn require_break_cx(&self, name: &str, span: Span) {
        let err_inside_of = |article, ty, closure_span| {
            struct_span_err!(self.sess, span, E0267, "`{}` inside of {} {}", name, article, ty)
                .span_label(span, format!("cannot `{}` inside of {} {}", name, article, ty))
                .span_label(closure_span, &format!("enclosing {}", ty))
                .emit();
        };

        match self.cx {
            LabeledBlock | Loop(_) => {}
            Closure(closure_span) => err_inside_of("a", "closure", closure_span),
            AsyncClosure(closure_span) => err_inside_of("an", "`async` block", closure_span),
            Normal | AnonConst => {
                struct_span_err!(self.sess, span, E0268, "`{}` outside of a loop", name)
                    .span_label(span, format!("cannot `{}` outside of a loop", name))
                    .emit();
            }
        }
    }

    fn require_label_in_labeled_block(
        &mut self,
        span: Span,
        label: &Destination,
        cf_type: &str,
    ) -> bool {
        if !span.is_desugaring(DesugaringKind::QuestionMark) && self.cx == LabeledBlock {
            if label.label.is_none() {
                struct_span_err!(
                    self.sess,
                    span,
                    E0695,
                    "unlabeled `{}` inside of a labeled block",
                    cf_type
                )
                .span_label(
                    span,
                    format!(
                        "`{}` statements that would diverge to or through \
                                a labeled block need to bear a label",
                        cf_type
                    ),
                )
                .emit();
                return true;
            }
        }
        false
    }
    fn emit_unlabled_cf_in_while_condition(&mut self, span: Span, cf_type: &str) {
        struct_span_err!(
            self.sess,
            span,
            E0590,
            "`break` or `continue` with no label in the condition of a `while` loop"
        )
        .span_label(span, format!("unlabeled `{}` in the condition of a `while` loop", cf_type))
        .emit();
    }
}
