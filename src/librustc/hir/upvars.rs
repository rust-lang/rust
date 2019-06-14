//! Upvar (closure capture) collection from cross-body HIR uses of `Res::Local`s.

use crate::hir::{self, HirId};
use crate::hir::def::Res;
use crate::hir::intravisit::{self, Visitor, NestedVisitorMap};
use crate::ty::TyCtxt;
use crate::ty::query::Providers;
use syntax_pos::Span;
use rustc_data_structures::fx::{FxIndexMap, FxHashSet};

pub fn provide(providers: &mut Providers<'_>) {
    providers.upvars = |tcx, def_id| {
        if !tcx.is_closure(def_id) {
            return None;
        }

        let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();
        let body = tcx.hir().body(tcx.hir().maybe_body_owned_by(hir_id)?);

        let mut local_collector = LocalCollector::default();
        local_collector.visit_body(body);

        let mut capture_collector = CaptureCollector {
            tcx,
            locals: &local_collector.locals,
            upvars: FxIndexMap::default(),
        };
        capture_collector.visit_body(body);

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

impl Visitor<'tcx> for LocalCollector {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_pat(&mut self, pat: &'tcx hir::Pat) {
        if let hir::PatKind::Binding(_, hir_id, ..) = pat.node {
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

impl Visitor<'tcx> for CaptureCollector<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_path(&mut self, path: &'tcx hir::Path, _: hir::HirId) {
        if let Res::Local(var_id) = path.res {
            self.visit_local_use(var_id, path.span);
        }

        intravisit::walk_path(self, path);
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        if let hir::ExprKind::Closure(..) = expr.node {
            let closure_def_id = self.tcx.hir().local_def_id_from_hir_id(expr.hir_id);
            if let Some(upvars) = self.tcx.upvars(closure_def_id) {
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
