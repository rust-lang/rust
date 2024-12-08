//! Upvar (closure capture) collection from cross-body HIR uses of `Res::Local`s.

use rustc_data_structures::fx::{FxHashSet, FxIndexMap};
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{self, HirId};
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

pub(crate) fn provide(providers: &mut Providers) {
    providers.upvars_mentioned = |tcx, def_id| {
        if !tcx.is_closure_like(def_id) {
            return None;
        }

        let local_def_id = def_id.expect_local();
        let body = tcx.hir().maybe_body_owned_by(local_def_id)?;

        let mut local_collector = LocalCollector::default();
        local_collector.visit_body(&body);

        let mut capture_collector = CaptureCollector {
            tcx,
            locals: &local_collector.locals,
            upvars: FxIndexMap::default(),
        };
        capture_collector.visit_body(&body);

        if !capture_collector.upvars.is_empty() {
            Some(tcx.arena.alloc(capture_collector.upvars))
        } else {
            None
        }
    };
}

#[derive(Default)]
struct LocalCollector {
    // FIXME(eddyb) perhaps use `ItemLocalId` instead?
    locals: FxHashSet<HirId>,
}

impl<'tcx> Visitor<'tcx> for LocalCollector {
    fn visit_pat(&mut self, pat: &'tcx hir::Pat<'tcx>) {
        if let hir::PatKind::Binding(_, hir_id, ..) = pat.kind {
            self.locals.insert(hir_id);
        }
        intravisit::walk_pat(self, pat);
    }
}

struct CaptureCollector<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    locals: &'a FxHashSet<HirId>,
    upvars: FxIndexMap<HirId, hir::Upvar>,
}

impl CaptureCollector<'_, '_> {
    fn visit_local_use(&mut self, var_id: HirId, span: Span) {
        if !self.locals.contains(&var_id) {
            self.upvars.entry(var_id).or_insert(hir::Upvar { span });
        }
    }
}

impl<'tcx> Visitor<'tcx> for CaptureCollector<'_, 'tcx> {
    fn visit_path(&mut self, path: &hir::Path<'tcx>, _: HirId) {
        if let Res::Local(var_id) = path.res {
            self.visit_local_use(var_id, path.span);
        }

        intravisit::walk_path(self, path);
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Closure(closure) = expr.kind {
            if let Some(upvars) = self.tcx.upvars_mentioned(closure.def_id) {
                // Every capture of a closure expression is a local in scope,
                // that is moved/copied/borrowed into the closure value, and
                // for this analysis they are like any other access to a local.
                //
                // E.g. in `|b| |c| (a, b, c)`, the upvars of the inner closure
                // are `a` and `b`, and while `a` is not directly used in the
                // outer closure, it needs to be an upvar there too, so that
                // the inner closure can take it (from the outer closure's env).
                for (&var_id, upvar) in upvars {
                    self.visit_local_use(var_id, upvar.span);
                }
            }
        }

        intravisit::walk_expr(self, expr);
    }
}
