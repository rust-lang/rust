use Context::*;

use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{Destination, Movability, Node};
use rustc_middle::hir::map::Map;
use rustc_middle::hir::nested_filter;
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_span::hygiene::DesugaringKind;
use rustc_span::Span;

use crate::errors::{
    BreakInsideAsyncBlock, BreakInsideClosure, BreakNonLoop, ContinueLabeledBlock, OutsideLoop,
    UnlabeledCfInWhileCondition, UnlabeledInLabeledBlock,
};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Context {
    Normal,
    Loop(hir::LoopSource),
    Closure(Span),
    AsyncClosure(Span),
    LabeledBlock,
    Constant,
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
        &mut CheckLoopVisitor { sess: &tcx.sess, hir_map: tcx.hir(), cx: Normal },
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
        self.with_context(Constant, |v| intravisit::walk_anon_const(v, c));
    }

    fn visit_inline_const(&mut self, c: &'hir hir::ConstBlock) {
        self.with_context(Constant, |v| intravisit::walk_inline_const(v, c));
    }

    fn visit_expr(&mut self, e: &'hir hir::Expr<'hir>) {
        match e.kind {
            hir::ExprKind::Loop(ref b, _, source, _) => {
                self.with_context(Loop(source), |v| v.visit_block(&b));
            }
            hir::ExprKind::Closure(&hir::Closure {
                ref fn_decl,
                body,
                fn_decl_span,
                movability,
                ..
            }) => {
                let cx = if let Some(Movability::Static) = movability {
                    AsyncClosure(fn_decl_span)
                } else {
                    Closure(fn_decl_span)
                };
                self.visit_fn_decl(&fn_decl);
                self.with_context(cx, |v| v.visit_nested_body(body));
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
                        self.sess.emit_err(UnlabeledCfInWhileCondition {
                            span: e.span,
                            cf_type: "break",
                        });
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
                            let suggestion = format!(
                                "break{}",
                                break_label
                                    .label
                                    .map_or_else(String::new, |l| format!(" {}", l.ident))
                            );
                            self.sess.emit_err(BreakNonLoop {
                                span: e.span,
                                head,
                                kind: kind.name(),
                                suggestion,
                                loop_label,
                                break_label: break_label.label,
                                break_expr_kind: &break_expr.kind,
                                break_expr_span: break_expr.span,
                            });
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
                            self.sess.emit_err(ContinueLabeledBlock {
                                span: e.span,
                                block_span: block.span,
                            });
                        }
                    }
                    Err(hir::LoopIdError::UnlabeledCfInWhileCondition) => {
                        self.sess.emit_err(UnlabeledCfInWhileCondition {
                            span: e.span,
                            cf_type: "continue",
                        });
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
        match self.cx {
            LabeledBlock | Loop(_) => {}
            Closure(closure_span) => {
                self.sess.emit_err(BreakInsideClosure { span, closure_span, name });
            }
            AsyncClosure(closure_span) => {
                self.sess.emit_err(BreakInsideAsyncBlock { span, closure_span, name });
            }
            Normal | Constant => {
                self.sess.emit_err(OutsideLoop { span, name, is_break: name == "break" });
            }
        }
    }

    fn require_label_in_labeled_block(
        &mut self,
        span: Span,
        label: &Destination,
        cf_type: &str,
    ) -> bool {
        if !span.is_desugaring(DesugaringKind::QuestionMark)
            && self.cx == LabeledBlock
            && label.label.is_none()
        {
            self.sess.emit_err(UnlabeledInLabeledBlock { span, cf_type });
            return true;
        }
        false
    }
}
