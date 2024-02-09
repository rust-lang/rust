use Context::*;

use rustc_hir as hir;
use rustc_hir::def_id::{LocalDefId, LocalModDefId};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{Destination, Node};
use rustc_middle::hir::nested_filter;
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_span::hygiene::DesugaringKind;
use rustc_span::{BytePos, Span};

use crate::errors::{
    BreakInsideAsyncBlock, BreakInsideClosure, BreakNonLoop, ContinueLabeledBlock, OutsideLoop,
    OutsideLoopSuggestion, UnlabeledCfInWhileCondition, UnlabeledInLabeledBlock,
};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Context {
    Normal,
    Fn,
    Loop(hir::LoopSource),
    Closure(Span),
    AsyncClosure(Span),
    UnlabeledBlock(Span),
    LabeledBlock,
    Constant,
}

#[derive(Copy, Clone)]
struct CheckLoopVisitor<'a, 'tcx> {
    sess: &'a Session,
    tcx: TyCtxt<'tcx>,
    cx: Context,
}

fn check_mod_loops(tcx: TyCtxt<'_>, module_def_id: LocalModDefId) {
    tcx.hir().visit_item_likes_in_module(
        module_def_id,
        &mut CheckLoopVisitor { sess: tcx.sess, tcx, cx: Normal },
    );
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { check_mod_loops, ..*providers };
}

impl<'a, 'hir> Visitor<'hir> for CheckLoopVisitor<'a, 'hir> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_anon_const(&mut self, c: &'hir hir::AnonConst) {
        self.with_context(Constant, |v| intravisit::walk_anon_const(v, c));
    }

    fn visit_inline_const(&mut self, c: &'hir hir::ConstBlock) {
        self.with_context(Constant, |v| intravisit::walk_inline_const(v, c));
    }

    fn visit_fn(
        &mut self,
        fk: hir::intravisit::FnKind<'hir>,
        fd: &'hir hir::FnDecl<'hir>,
        b: hir::BodyId,
        _: Span,
        id: LocalDefId,
    ) {
        self.with_context(Fn, |v| intravisit::walk_fn(v, fk, fd, b, id));
    }

    fn visit_trait_item(&mut self, trait_item: &'hir hir::TraitItem<'hir>) {
        self.with_context(Fn, |v| intravisit::walk_trait_item(v, trait_item));
    }

    fn visit_impl_item(&mut self, impl_item: &'hir hir::ImplItem<'hir>) {
        self.with_context(Fn, |v| intravisit::walk_impl_item(v, impl_item));
    }

    fn visit_expr(&mut self, e: &'hir hir::Expr<'hir>) {
        match e.kind {
            hir::ExprKind::Loop(ref b, _, source, _) => {
                self.with_context(Loop(source), |v| v.visit_block(b));
            }
            hir::ExprKind::Closure(&hir::Closure {
                ref fn_decl, body, fn_decl_span, kind, ..
            }) => {
                // FIXME(coroutines): This doesn't handle coroutines correctly
                let cx = match kind {
                    hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                        hir::CoroutineDesugaring::Async,
                        hir::CoroutineSource::Block,
                    )) => AsyncClosure(fn_decl_span),
                    _ => Closure(fn_decl_span),
                };
                self.visit_fn_decl(fn_decl);
                self.with_context(cx, |v| v.visit_nested_body(body));
            }
            hir::ExprKind::Block(ref b, Some(_label)) => {
                self.with_context(LabeledBlock, |v| v.visit_block(b));
            }
            hir::ExprKind::Block(ref b, None) if matches!(self.cx, Fn) => {
                self.with_context(Normal, |v| v.visit_block(b));
            }
            hir::ExprKind::Block(ref b, None)
                if matches!(self.cx, Normal | Constant | UnlabeledBlock(_)) =>
            {
                self.with_context(UnlabeledBlock(b.span.shrink_to_lo()), |v| v.visit_block(b));
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
                        self.sess.dcx().emit_err(UnlabeledCfInWhileCondition {
                            span: e.span,
                            cf_type: "break",
                        });
                        None
                    }
                    Err(hir::LoopIdError::UnresolvedLabel) => None,
                };

                if let Some(Node::Block(_)) = loop_id.map(|id| self.tcx.hir_node(id)) {
                    return;
                }

                if let Some(break_expr) = opt_expr {
                    let (head, loop_label, loop_kind) = if let Some(loop_id) = loop_id {
                        match self.tcx.hir().expect_expr(loop_id).kind {
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
                            self.sess.dcx().emit_err(BreakNonLoop {
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

                let sp_lo = e.span.with_lo(e.span.lo() + BytePos("break".len() as u32));
                let label_sp = match break_label.label {
                    Some(label) => sp_lo.with_hi(label.ident.span.hi()),
                    None => sp_lo.shrink_to_lo(),
                };
                self.require_break_cx("break", e.span, label_sp);
            }
            hir::ExprKind::Continue(destination) => {
                self.require_label_in_labeled_block(e.span, &destination, "continue");

                match destination.target_id {
                    Ok(loop_id) => {
                        if let Node::Block(block) = self.tcx.hir_node(loop_id) {
                            self.sess.dcx().emit_err(ContinueLabeledBlock {
                                span: e.span,
                                block_span: block.span,
                            });
                        }
                    }
                    Err(hir::LoopIdError::UnlabeledCfInWhileCondition) => {
                        self.sess.dcx().emit_err(UnlabeledCfInWhileCondition {
                            span: e.span,
                            cf_type: "continue",
                        });
                    }
                    Err(_) => {}
                }
                self.require_break_cx("continue", e.span, e.span)
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

    fn require_break_cx(&self, name: &str, span: Span, break_span: Span) {
        let is_break = name == "break";
        match self.cx {
            LabeledBlock | Loop(_) => {}
            Closure(closure_span) => {
                self.sess.dcx().emit_err(BreakInsideClosure { span, closure_span, name });
            }
            AsyncClosure(closure_span) => {
                self.sess.dcx().emit_err(BreakInsideAsyncBlock { span, closure_span, name });
            }
            UnlabeledBlock(block_span) if is_break && block_span.eq_ctxt(break_span) => {
                let suggestion = Some(OutsideLoopSuggestion { block_span, break_span });
                self.sess.dcx().emit_err(OutsideLoop { span, name, is_break, suggestion });
            }
            Normal | Constant | Fn | UnlabeledBlock(_) => {
                self.sess.dcx().emit_err(OutsideLoop { span, name, is_break, suggestion: None });
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
            self.sess.dcx().emit_err(UnlabeledInLabeledBlock { span, cf_type });
            return true;
        }
        false
    }
}
