use std::collections::BTreeMap;
use std::fmt;

use Context::*;
use rustc_hir as hir;
use rustc_hir::def_id::{LocalDefId, LocalModDefId};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{Destination, Node};
use rustc_middle::hir::nested_filter;
use rustc_middle::query::Providers;
use rustc_middle::span_bug;
use rustc_middle::ty::TyCtxt;
use rustc_span::hygiene::DesugaringKind;
use rustc_span::{BytePos, Span};

use crate::errors::{
    BreakInsideClosure, BreakInsideCoroutine, BreakNonLoop, ContinueLabeledBlock, OutsideLoop,
    OutsideLoopSuggestion, UnlabeledCfInWhileCondition, UnlabeledInLabeledBlock,
};

/// The context in which a block is encountered.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Context {
    Normal,
    Fn,
    Loop(hir::LoopSource),
    Closure(Span),
    Coroutine {
        coroutine_span: Span,
        kind: hir::CoroutineDesugaring,
        source: hir::CoroutineSource,
    },
    UnlabeledBlock(Span),
    UnlabeledIfBlock(Span),
    LabeledBlock,
    /// E.g. The labeled block inside `['_'; 'block: { break 'block 1 + 2; }]`.
    AnonConst,
    /// E.g. `const { ... }`.
    ConstBlock,
}

#[derive(Clone)]
struct BlockInfo {
    name: String,
    spans: Vec<Span>,
    suggs: Vec<Span>,
}

#[derive(PartialEq)]
enum BreakContextKind {
    Break,
    Continue,
}

impl fmt::Display for BreakContextKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BreakContextKind::Break => "break",
            BreakContextKind::Continue => "continue",
        }
        .fmt(f)
    }
}

#[derive(Clone)]
struct CheckLoopVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    // Keep track of a stack of contexts, so that suggestions
    // are not made for contexts where it would be incorrect,
    // such as adding a label for an `if`.
    // e.g. `if 'foo: {}` would be incorrect.
    cx_stack: Vec<Context>,
    block_breaks: BTreeMap<Span, BlockInfo>,
}

fn check_mod_loops(tcx: TyCtxt<'_>, module_def_id: LocalModDefId) {
    let mut check =
        CheckLoopVisitor { tcx, cx_stack: vec![Normal], block_breaks: Default::default() };
    tcx.hir().visit_item_likes_in_module(module_def_id, &mut check);
    check.report_outside_loop_error();
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { check_mod_loops, ..*providers };
}

impl<'hir> Visitor<'hir> for CheckLoopVisitor<'hir> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_anon_const(&mut self, c: &'hir hir::AnonConst) {
        self.with_context(AnonConst, |v| intravisit::walk_anon_const(v, c));
    }

    fn visit_inline_const(&mut self, c: &'hir hir::ConstBlock) {
        self.with_context(ConstBlock, |v| intravisit::walk_inline_const(v, c));
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
            hir::ExprKind::If(cond, then, else_opt) => {
                self.visit_expr(cond);

                let get_block = |ck_loop: &CheckLoopVisitor<'hir>,
                                 expr: &hir::Expr<'hir>|
                 -> Option<&hir::Block<'hir>> {
                    if let hir::ExprKind::Block(b, None) = expr.kind
                        && matches!(
                            ck_loop.cx_stack.last(),
                            Some(&Normal)
                                | Some(&AnonConst)
                                | Some(&UnlabeledBlock(_))
                                | Some(&UnlabeledIfBlock(_))
                        )
                    {
                        Some(b)
                    } else {
                        None
                    }
                };

                if let Some(b) = get_block(self, then) {
                    self.with_context(UnlabeledIfBlock(b.span.shrink_to_lo()), |v| {
                        v.visit_block(b)
                    });
                } else {
                    self.visit_expr(then);
                }

                if let Some(else_expr) = else_opt {
                    if let Some(b) = get_block(self, else_expr) {
                        self.with_context(UnlabeledIfBlock(b.span.shrink_to_lo()), |v| {
                            v.visit_block(b)
                        });
                    } else {
                        self.visit_expr(else_expr);
                    }
                }
            }
            hir::ExprKind::Loop(ref b, _, source, _) => {
                self.with_context(Loop(source), |v| v.visit_block(b));
            }
            hir::ExprKind::Closure(&hir::Closure {
                ref fn_decl, body, fn_decl_span, kind, ..
            }) => {
                let cx = match kind {
                    hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(kind, source)) => {
                        Coroutine { coroutine_span: fn_decl_span, kind, source }
                    }
                    _ => Closure(fn_decl_span),
                };
                self.visit_fn_decl(fn_decl);
                self.with_context(cx, |v| v.visit_nested_body(body));
            }
            hir::ExprKind::Block(ref b, Some(_label)) => {
                self.with_context(LabeledBlock, |v| v.visit_block(b));
            }
            hir::ExprKind::Block(ref b, None)
                if matches!(self.cx_stack.last(), Some(&Fn) | Some(&ConstBlock)) =>
            {
                self.with_context(Normal, |v| v.visit_block(b));
            }
            hir::ExprKind::Block(
                ref b @ hir::Block { rules: hir::BlockCheckMode::DefaultBlock, .. },
                None,
            ) if matches!(
                self.cx_stack.last(),
                Some(&Normal) | Some(&AnonConst) | Some(&UnlabeledBlock(_))
            ) =>
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
                        self.tcx.dcx().emit_err(UnlabeledCfInWhileCondition {
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
                            self.tcx.dcx().emit_err(BreakNonLoop {
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
                self.require_break_cx(
                    BreakContextKind::Break,
                    e.span,
                    label_sp,
                    self.cx_stack.len() - 1,
                );
            }
            hir::ExprKind::Continue(destination) => {
                self.require_label_in_labeled_block(e.span, &destination, "continue");

                match destination.target_id {
                    Ok(loop_id) => {
                        if let Node::Block(block) = self.tcx.hir_node(loop_id) {
                            self.tcx.dcx().emit_err(ContinueLabeledBlock {
                                span: e.span,
                                block_span: block.span,
                            });
                        }
                    }
                    Err(hir::LoopIdError::UnlabeledCfInWhileCondition) => {
                        self.tcx.dcx().emit_err(UnlabeledCfInWhileCondition {
                            span: e.span,
                            cf_type: "continue",
                        });
                    }
                    Err(_) => {}
                }
                self.require_break_cx(
                    BreakContextKind::Continue,
                    e.span,
                    e.span,
                    self.cx_stack.len() - 1,
                )
            }
            _ => intravisit::walk_expr(self, e),
        }
    }
}

impl<'hir> CheckLoopVisitor<'hir> {
    fn with_context<F>(&mut self, cx: Context, f: F)
    where
        F: FnOnce(&mut CheckLoopVisitor<'hir>),
    {
        self.cx_stack.push(cx);
        f(self);
        self.cx_stack.pop();
    }

    fn require_break_cx(
        &mut self,
        br_cx_kind: BreakContextKind,
        span: Span,
        break_span: Span,
        cx_pos: usize,
    ) {
        match self.cx_stack[cx_pos] {
            LabeledBlock | Loop(_) => {}
            Closure(closure_span) => {
                self.tcx.dcx().emit_err(BreakInsideClosure {
                    span,
                    closure_span,
                    name: &br_cx_kind.to_string(),
                });
            }
            Coroutine { coroutine_span, kind, source } => {
                let kind = match kind {
                    hir::CoroutineDesugaring::Async => "async",
                    hir::CoroutineDesugaring::Gen => "gen",
                    hir::CoroutineDesugaring::AsyncGen => "async gen",
                };
                let source = match source {
                    hir::CoroutineSource::Block => "block",
                    hir::CoroutineSource::Closure => "closure",
                    hir::CoroutineSource::Fn => "function",
                };
                self.tcx.dcx().emit_err(BreakInsideCoroutine {
                    span,
                    coroutine_span,
                    name: &br_cx_kind.to_string(),
                    kind,
                    source,
                });
            }
            UnlabeledBlock(block_span)
                if br_cx_kind == BreakContextKind::Break && block_span.eq_ctxt(break_span) =>
            {
                let block = self.block_breaks.entry(block_span).or_insert_with(|| BlockInfo {
                    name: br_cx_kind.to_string(),
                    spans: vec![],
                    suggs: vec![],
                });
                block.spans.push(span);
                block.suggs.push(break_span);
            }
            UnlabeledIfBlock(_) if br_cx_kind == BreakContextKind::Break => {
                self.require_break_cx(br_cx_kind, span, break_span, cx_pos - 1);
            }
            Normal | AnonConst | Fn | UnlabeledBlock(_) | UnlabeledIfBlock(_) | ConstBlock => {
                self.tcx.dcx().emit_err(OutsideLoop {
                    spans: vec![span],
                    name: &br_cx_kind.to_string(),
                    is_break: br_cx_kind == BreakContextKind::Break,
                    suggestion: None,
                });
            }
        }
    }

    fn require_label_in_labeled_block(
        &self,
        span: Span,
        label: &Destination,
        cf_type: &str,
    ) -> bool {
        if !span.is_desugaring(DesugaringKind::QuestionMark)
            && self.cx_stack.last() == Some(&LabeledBlock)
            && label.label.is_none()
        {
            self.tcx.dcx().emit_err(UnlabeledInLabeledBlock { span, cf_type });
            return true;
        }
        false
    }

    fn report_outside_loop_error(&self) {
        for (s, block) in &self.block_breaks {
            self.tcx.dcx().emit_err(OutsideLoop {
                spans: block.spans.clone(),
                name: &block.name,
                is_break: true,
                suggestion: Some(OutsideLoopSuggestion {
                    block_span: *s,
                    break_spans: block.suggs.clone(),
                }),
            });
        }
    }
}
