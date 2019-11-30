//! This pass checks HIR bodies that may be evaluated at compile-time (e.g., `const`, `static`,
//! `const fn`) for structured control flow (e.g. `if`, `while`), which is forbidden in a const
//! context.
//!
//! By the time the MIR const-checker runs, these high-level constructs have been lowered to
//! control-flow primitives (e.g., `Goto`, `SwitchInt`), making it tough to properly attribute
//! errors. We still look for those primitives in the MIR const-checker to ensure nothing slips
//! through, but errors for structured control flow in a `const` should be emitted here.

use rustc::hir::def_id::DefId;
use rustc::hir::intravisit::{Visitor, NestedVisitorMap};
use rustc::hir::map::Map;
use rustc::hir;
use rustc::ty::TyCtxt;
use rustc::ty::query::Providers;
use rustc_feature::Features;
use syntax::ast::Mutability;
use syntax::feature_gate::feature_err;
use syntax::span_err;
use syntax_pos::{sym, Span};
use rustc_error_codes::*;

use std::fmt;

/// An expression that is not *always* legal in a const context.
#[derive(Clone, Copy)]
enum NonConstExpr {
    Loop(hir::LoopSource),
    Match(hir::MatchSource),
}

impl NonConstExpr {
    fn name(self) -> &'static str {
        match self {
            Self::Loop(src) => src.name(),
            Self::Match(src) => src.name(),
        }
    }

    /// Returns `true` if all feature gates required to enable this expression are turned on, or
    /// `None` if there is no feature gate corresponding to this expression.
    fn is_feature_gate_enabled(self, features: &Features) -> Option<bool> {
        use hir::MatchSource::*;
        match self {
            | Self::Match(Normal)
            | Self::Match(IfDesugar { .. })
            | Self::Match(IfLetDesugar { .. })
            => Some(features.const_if_match),

            _ => None,
        }
    }
}

#[derive(Copy, Clone)]
enum ConstKind {
    Static,
    StaticMut,
    ConstFn,
    Const,
    AnonConst,
}

impl ConstKind {
    fn for_body(body: &hir::Body, hir_map: &Map<'_>) -> Option<Self> {
        let is_const_fn = |id| hir_map.fn_sig_by_hir_id(id).unwrap().header.is_const();

        let owner = hir_map.body_owner(body.id());
        let const_kind = match hir_map.body_owner_kind(owner) {
            hir::BodyOwnerKind::Const => Self::Const,
            hir::BodyOwnerKind::Static(Mutability::Mutable) => Self::StaticMut,
            hir::BodyOwnerKind::Static(Mutability::Immutable) => Self::Static,

            hir::BodyOwnerKind::Fn if is_const_fn(owner) => Self::ConstFn,
            hir::BodyOwnerKind::Fn | hir::BodyOwnerKind::Closure => return None,
        };

        Some(const_kind)
    }
}

impl fmt::Display for ConstKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Static => "static",
            Self::StaticMut => "static mut",
            Self::Const | Self::AnonConst => "const",
            Self::ConstFn => "const fn",
        };

        write!(f, "{}", s)
    }
}

fn check_mod_const_bodies(tcx: TyCtxt<'_>, module_def_id: DefId) {
    let mut vis = CheckConstVisitor::new(tcx);
    tcx.hir().visit_item_likes_in_module(module_def_id, &mut vis.as_deep_visitor());
}

pub(crate) fn provide(providers: &mut Providers<'_>) {
    *providers = Providers {
        check_mod_const_bodies,
        ..*providers
    };
}

#[derive(Copy, Clone)]
struct CheckConstVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    const_kind: Option<ConstKind>,
}

impl<'tcx> CheckConstVisitor<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Self {
        CheckConstVisitor {
            tcx,
            const_kind: None,
        }
    }

    /// Emits an error when an unsupported expression is found in a const context.
    fn const_check_violated(&self, expr: NonConstExpr, span: Span) {
        match expr.is_feature_gate_enabled(self.tcx.features()) {
            // Don't emit an error if the user has enabled the requisite feature gates.
            Some(true) => return,

            // Users of `-Zunleash-the-miri-inside-of-you` must use feature gates when possible.
            None if self.tcx.sess.opts.debugging_opts.unleash_the_miri_inside_of_you => {
                self.tcx.sess.span_warn(span, "skipping const checks");
                return;
            }

            _ => {}
        }

        let const_kind = self.const_kind
            .expect("`const_check_violated` may only be called inside a const context");

        let msg = format!("`{}` is not allowed in a `{}`", expr.name(), const_kind);
        match expr {
            | NonConstExpr::Match(hir::MatchSource::Normal)
            | NonConstExpr::Match(hir::MatchSource::IfDesugar { .. })
            | NonConstExpr::Match(hir::MatchSource::IfLetDesugar { .. })
            => feature_err(&self.tcx.sess.parse_sess, sym::const_if_match, span, &msg).emit(),

            _ => span_err!(self.tcx.sess, span, E0744, "{}", msg),
        }
    }

    /// Saves the parent `const_kind` before calling `f` and restores it afterwards.
    fn recurse_into(&mut self, kind: Option<ConstKind>, f: impl FnOnce(&mut Self)) {
        let parent_kind = self.const_kind;
        self.const_kind = kind;
        f(self);
        self.const_kind = parent_kind;
    }
}

impl<'tcx> Visitor<'tcx> for CheckConstVisitor<'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.tcx.hir())
    }

    fn visit_anon_const(&mut self, anon: &'tcx hir::AnonConst) {
        let kind = Some(ConstKind::AnonConst);
        self.recurse_into(kind, |this| hir::intravisit::walk_anon_const(this, anon));
    }

    fn visit_body(&mut self, body: &'tcx hir::Body) {
        let kind = ConstKind::for_body(body, self.tcx.hir());
        self.recurse_into(kind, |this| hir::intravisit::walk_body(this, body));
    }

    fn visit_expr(&mut self, e: &'tcx hir::Expr) {
        match &e.kind {
            // Skip the following checks if we are not currently in a const context.
            _ if self.const_kind.is_none() => {}

            hir::ExprKind::Loop(_, _, source) => {
                self.const_check_violated(NonConstExpr::Loop(*source), e.span);
            }

            hir::ExprKind::Match(_, _, source) => {
                let non_const_expr = match source {
                    // These are handled by `ExprKind::Loop` above.
                    | hir::MatchSource::WhileDesugar
                    | hir::MatchSource::WhileLetDesugar
                    | hir::MatchSource::ForLoopDesugar
                    => None,

                    _ => Some(NonConstExpr::Match(*source)),
                };

                if let Some(expr) = non_const_expr {
                    self.const_check_violated(expr, e.span);
                }
            }

            _ => {},
        }

        hir::intravisit::walk_expr(self, e);
    }
}
