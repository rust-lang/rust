use Context::*;

use rustc::session::Session;

use rustc::ty::query::Providers;
use rustc::ty::TyCtxt;
use rustc::hir::def_id::DefId;
use rustc::hir::map::Map;
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::hir::{self, Node, Destination};
use syntax::struct_span_err;
use syntax_pos::Span;
use errors::Applicability;

#[derive(Clone, Copy, Debug, PartialEq)]
enum LoopKind {
    Loop(hir::LoopSource),
    WhileLoop,
}

impl LoopKind {
    fn name(self) -> &'static str {
        match self {
            LoopKind::Loop(hir::LoopSource::Loop) => "loop",
            LoopKind::Loop(hir::LoopSource::WhileLet) => "while let",
            LoopKind::Loop(hir::LoopSource::ForLoop) => "for",
            LoopKind::WhileLoop => "while",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Context {
    Normal,
    Loop(LoopKind),
    Closure,
    LabeledBlock,
    AnonConst,
}

#[derive(Copy, Clone)]
struct CheckLoopVisitor<'a, 'hir> {
    sess: &'a Session,
    hir_map: &'a Map<'hir>,
    cx: Context,
}

fn check_mod_loops(tcx: TyCtxt<'_>, module_def_id: DefId) {
    tcx.hir().visit_item_likes_in_module(module_def_id, &mut CheckLoopVisitor {
        sess: &tcx.sess,
        hir_map: &tcx.hir(),
        cx: Normal,
    }.as_deep_visitor());
}

pub(crate) fn provide(providers: &mut Providers<'_>) {
    *providers = Providers {
        check_mod_loops,
        ..*providers
    };
}

impl<'a, 'hir> Visitor<'hir> for CheckLoopVisitor<'a, 'hir> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'hir> {
        NestedVisitorMap::OnlyBodies(&self.hir_map)
    }

    fn visit_anon_const(&mut self, c: &'hir hir::AnonConst) {
        self.with_context(AnonConst, |v| intravisit::walk_anon_const(v, c));
    }

    fn visit_expr(&mut self, e: &'hir hir::Expr) {
        match e.node {
            hir::ExprKind::While(ref e, ref b, _) => {
                self.with_context(Loop(LoopKind::WhileLoop), |v| {
                    v.visit_expr(&e);
                    v.visit_block(&b);
                });
            }
            hir::ExprKind::Loop(ref b, _, source) => {
                self.with_context(Loop(LoopKind::Loop(source)), |v| v.visit_block(&b));
            }
            hir::ExprKind::Closure(_, ref function_decl, b, _, _) => {
                self.visit_fn_decl(&function_decl);
                self.with_context(Closure, |v| v.visit_nested_body(b));
            }
            hir::ExprKind::Block(ref b, Some(_label)) => {
                self.with_context(LabeledBlock, |v| v.visit_block(&b));
            }
            hir::ExprKind::Break(label, ref opt_expr) => {
                opt_expr.as_ref().map(|e| self.visit_expr(e));

                if self.require_label_in_labeled_block(e.span, &label, "break") {
                    // If we emitted an error about an unlabeled break in a labeled
                    // block, we don't need any further checking for this break any more
                    return;
                }

                let loop_id = match label.target_id.into() {
                    Ok(loop_id) => loop_id,
                    Err(hir::LoopIdError::OutsideLoopScope) => hir::DUMMY_HIR_ID,
                    Err(hir::LoopIdError::UnlabeledCfInWhileCondition) => {
                        self.emit_unlabled_cf_in_while_condition(e.span, "break");
                        hir::DUMMY_HIR_ID
                    },
                    Err(hir::LoopIdError::UnresolvedLabel) => hir::DUMMY_HIR_ID,
                };

                if loop_id != hir::DUMMY_HIR_ID {
                    if let Node::Block(_) = self.hir_map.find(loop_id).unwrap() {
                        return
                    }
                }

                if opt_expr.is_some() {
                    let loop_kind = if loop_id == hir::DUMMY_HIR_ID {
                        None
                    } else {
                        Some(match self.hir_map.expect_expr(loop_id).node {
                            hir::ExprKind::While(..) => LoopKind::WhileLoop,
                            hir::ExprKind::Loop(_, _, source) => LoopKind::Loop(source),
                            ref r => span_bug!(e.span,
                                               "break label resolved to a non-loop: {:?}", r),
                        })
                    };
                    match loop_kind {
                        None |
                        Some(LoopKind::Loop(hir::LoopSource::Loop)) => (),
                        Some(kind) => {
                            struct_span_err!(self.sess, e.span, E0571,
                                             "`break` with value from a `{}` loop",
                                             kind.name())
                                .span_label(e.span,
                                            "can only break with a value inside \
                                            `loop` or breakable block")
                                .span_suggestion(
                                    e.span,
                                    &format!(
                                        "instead, use `break` on its own \
                                        without a value inside this `{}` loop",
                                        kind.name()
                                    ),
                                    "break".to_string(),
                                    Applicability::MaybeIncorrect,
                                )
                                .emit();
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
                            struct_span_err!(self.sess, e.span, E0696,
                                            "`continue` pointing to a labeled block")
                                .span_label(e.span,
                                            "labeled blocks cannot be `continue`'d")
                                .span_note(block.span,
                                            "labeled block the continue points to")
                                .emit();
                        }
                    }
                    Err(hir::LoopIdError::UnlabeledCfInWhileCondition) => {
                        self.emit_unlabled_cf_in_while_condition(e.span, "continue");
                    }
                    Err(_) => {}
                }
                self.require_break_cx("continue", e.span)
            },
            _ => intravisit::walk_expr(self, e),
        }
    }
}

impl<'a, 'hir> CheckLoopVisitor<'a, 'hir> {
    fn with_context<F>(&mut self, cx: Context, f: F)
        where F: FnOnce(&mut CheckLoopVisitor<'a, 'hir>)
    {
        let old_cx = self.cx;
        self.cx = cx;
        f(self);
        self.cx = old_cx;
    }

    fn require_break_cx(&self, name: &str, span: Span) {
        match self.cx {
            LabeledBlock | Loop(_) => {}
            Closure => {
                struct_span_err!(self.sess, span, E0267, "`{}` inside of a closure", name)
                .span_label(span, "cannot break inside of a closure")
                .emit();
            }
            Normal | AnonConst => {
                struct_span_err!(self.sess, span, E0268, "`{}` outside of loop", name)
                .span_label(span, "cannot break outside of a loop")
                .emit();
            }
        }
    }

    fn require_label_in_labeled_block(&mut self, span: Span, label: &Destination, cf_type: &str)
        -> bool
    {
        if self.cx == LabeledBlock {
            if label.label.is_none() {
                struct_span_err!(self.sess, span, E0695,
                                "unlabeled `{}` inside of a labeled block", cf_type)
                    .span_label(span,
                                format!("`{}` statements that would diverge to or through \
                                a labeled block need to bear a label", cf_type))
                    .emit();
                return true;
            }
        }
        return false;
    }
    fn emit_unlabled_cf_in_while_condition(&mut self, span: Span, cf_type: &str) {
        struct_span_err!(self.sess, span, E0590,
                         "`break` or `continue` with no label in the condition of a `while` loop")
            .span_label(span,
                        format!("unlabeled `{}` in the condition of a `while` loop", cf_type))
            .emit();
    }
}
