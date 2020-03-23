//! This pass checks HIR bodies that may be evaluated at compile-time (e.g., `const`, `static`,
//! `const fn`) for structured control flow (e.g. `if`, `while`), which is forbidden in a const
//! context.
//!
//! By the time the MIR const-checker runs, these high-level constructs have been lowered to
//! control-flow primitives (e.g., `Goto`, `SwitchInt`), making it tough to properly attribute
//! errors. We still look for those primitives in the MIR const-checker to ensure nothing slips
//! through, but errors for structured control flow in a `const` should be emitted here.

use rustc::hir::map::Map;
use rustc::ty::query::Providers;
use rustc::ty::TyCtxt;
use rustc_ast::ast::Mutability;
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_session::config::nightly_options;
use rustc_session::parse::feature_err;
use rustc_span::{sym, Span, Symbol};

use std::fmt;

/// An expression that is not *always* legal in a const context.
#[derive(Clone, Copy)]
enum NonConstExpr {
    Loop(hir::LoopSource),
    Match(hir::MatchSource),
    OrPattern,
}

impl NonConstExpr {
    fn name(self) -> String {
        match self {
            Self::Loop(src) => format!("`{}`", src.name()),
            Self::Match(src) => format!("`{}`", src.name()),
            Self::OrPattern => "or-pattern".to_string(),
        }
    }

    fn required_feature_gates(self) -> Option<&'static [Symbol]> {
        use hir::LoopSource::*;
        use hir::MatchSource::*;

        let gates: &[_] = match self {
            Self::Match(Normal)
            | Self::Match(IfDesugar { .. })
            | Self::Match(IfLetDesugar { .. })
            | Self::OrPattern => &[sym::const_if_match],

            Self::Loop(Loop) => &[sym::const_loop],

            Self::Loop(While)
            | Self::Loop(WhileLet)
            | Self::Match(WhileDesugar)
            | Self::Match(WhileLetDesugar) => &[sym::const_loop, sym::const_if_match],

            // A `for` loop's desugaring contains a call to `IntoIterator::into_iter`,
            // so they are not yet allowed with `#![feature(const_loop)]`.
            _ => return None,
        };

        Some(gates)
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
    fn for_body(body: &hir::Body<'_>, hir_map: Map<'_>) -> Option<Self> {
        let is_const_fn = |id| hir_map.fn_sig_by_hir_id(id).unwrap().header.is_const();

        let owner = hir_map.body_owner(body.id());
        let const_kind = match hir_map.body_owner_kind(owner) {
            hir::BodyOwnerKind::Const => Self::Const,
            hir::BodyOwnerKind::Static(Mutability::Mut) => Self::StaticMut,
            hir::BodyOwnerKind::Static(Mutability::Not) => Self::Static,

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
    *providers = Providers { check_mod_const_bodies, ..*providers };
}

#[derive(Copy, Clone)]
struct CheckConstVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    const_kind: Option<ConstKind>,
}

impl<'tcx> CheckConstVisitor<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Self {
        CheckConstVisitor { tcx, const_kind: None }
    }

    /// Emits an error when an unsupported expression is found in a const context.
    fn const_check_violated(&self, expr: NonConstExpr, span: Span) {
        let features = self.tcx.features();
        let required_gates = expr.required_feature_gates();
        match required_gates {
            // Don't emit an error if the user has enabled the requisite feature gates.
            Some(gates) if gates.iter().all(|&g| features.enabled(g)) => return,

            // `-Zunleash-the-miri-inside-of-you` only works for expressions that don't have a
            // corresponding feature gate. This encourages nightly users to use feature gates when
            // possible.
            None if self.tcx.sess.opts.debugging_opts.unleash_the_miri_inside_of_you => {
                self.tcx.sess.span_warn(span, "skipping const checks");
                return;
            }

            _ => {}
        }

        let const_kind = self
            .const_kind
            .expect("`const_check_violated` may only be called inside a const context");
        let msg = format!("{} is not allowed in a `{}`", expr.name(), const_kind);

        let required_gates = required_gates.unwrap_or(&[]);
        let missing_gates: Vec<_> =
            required_gates.iter().copied().filter(|&g| !features.enabled(g)).collect();

        match missing_gates.as_slice() {
            &[] => struct_span_err!(self.tcx.sess, span, E0744, "{}", msg).emit(),

            // If the user enabled `#![feature(const_loop)]` but not `#![feature(const_if_match)]`,
            // explain why their `while` loop is being rejected.
            &[gate @ sym::const_if_match] if required_gates.contains(&sym::const_loop) => {
                feature_err(&self.tcx.sess.parse_sess, gate, span, &msg)
                    .note(
                        "`#![feature(const_loop)]` alone is not sufficient, \
                           since this loop expression contains an implicit conditional",
                    )
                    .emit();
            }

            &[missing_primary, ref missing_secondary @ ..] => {
                let mut err = feature_err(&self.tcx.sess.parse_sess, missing_primary, span, &msg);

                // If multiple feature gates would be required to enable this expression, include
                // them as help messages. Don't emit a separate error for each missing feature gate.
                //
                // FIXME(ecstaticmorse): Maybe this could be incorporated into `feature_err`? This
                // is a pretty narrow case, however.
                if nightly_options::is_nightly_build() {
                    for gate in missing_secondary {
                        let note = format!(
                            "add `#![feature({})]` to the crate attributes to enable",
                            gate,
                        );
                        err.help(&note);
                    }
                }

                err.emit();
            }
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
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.tcx.hir())
    }

    fn visit_anon_const(&mut self, anon: &'tcx hir::AnonConst) {
        let kind = Some(ConstKind::AnonConst);
        self.recurse_into(kind, |this| intravisit::walk_anon_const(this, anon));
    }

    fn visit_body(&mut self, body: &'tcx hir::Body<'tcx>) {
        let kind = ConstKind::for_body(body, self.tcx.hir());
        self.recurse_into(kind, |this| intravisit::walk_body(this, body));
    }

    fn visit_pat(&mut self, p: &'tcx hir::Pat<'tcx>) {
        if self.const_kind.is_some() {
            if let hir::PatKind::Or { .. } = p.kind {
                self.const_check_violated(NonConstExpr::OrPattern, p.span);
            }
        }
        intravisit::walk_pat(self, p)
    }

    fn visit_expr(&mut self, e: &'tcx hir::Expr<'tcx>) {
        match &e.kind {
            // Skip the following checks if we are not currently in a const context.
            _ if self.const_kind.is_none() => {}

            hir::ExprKind::Loop(_, _, source) => {
                self.const_check_violated(NonConstExpr::Loop(*source), e.span);
            }

            hir::ExprKind::Match(_, _, source) => {
                let non_const_expr = match source {
                    // These are handled by `ExprKind::Loop` above.
                    hir::MatchSource::WhileDesugar
                    | hir::MatchSource::WhileLetDesugar
                    | hir::MatchSource::ForLoopDesugar => None,

                    _ => Some(NonConstExpr::Match(*source)),
                };

                if let Some(expr) = non_const_expr {
                    self.const_check_violated(expr, e.span);
                }
            }

            _ => {}
        }

        intravisit::walk_expr(self, e);
    }
}
