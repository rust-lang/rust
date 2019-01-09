use rustc::hir::intravisit::{Visitor, NestedVisitorMap};
use rustc::hir::{self, HirId};
use rustc::lint::builtin::UNUSED_MUT;
use rustc::ty;
use rustc::util::nodemap::{FxHashMap, FxHashSet};
use errors::Applicability;
use std::slice;
use syntax::ptr::P;

use borrowck::BorrowckCtxt;

pub fn check<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>, body: &'tcx hir::Body) {
    let mut used_mut = bccx.used_mut_nodes.borrow().clone();
    UsedMutFinder {
        bccx,
        set: &mut used_mut,
    }.visit_expr(&body.value);
    let mut cx = UnusedMutCx { bccx, used_mut };
    for arg in body.arguments.iter() {
        cx.check_unused_mut_pat(slice::from_ref(&arg.pat));
    }
    cx.visit_expr(&body.value);
}

struct UsedMutFinder<'a, 'tcx: 'a> {
    bccx: &'a BorrowckCtxt<'a, 'tcx>,
    set: &'a mut FxHashSet<HirId>,
}

struct UnusedMutCx<'a, 'tcx: 'a> {
    bccx: &'a BorrowckCtxt<'a, 'tcx>,
    used_mut: FxHashSet<HirId>,
}

impl<'a, 'tcx> UnusedMutCx<'a, 'tcx> {
    fn check_unused_mut_pat(&self, pats: &[P<hir::Pat>]) {
        let tcx = self.bccx.tcx;
        let mut mutables: FxHashMap<_, Vec<_>> = Default::default();
        for p in pats {
            p.each_binding(|_, hir_id, span, ident| {
                // Skip anything that looks like `_foo`
                if ident.as_str().starts_with("_") {
                    return;
                }

                // Skip anything that looks like `&foo` or `&mut foo`, only look
                // for by-value bindings
                if let Some(&bm) = self.bccx.tables.pat_binding_modes().get(hir_id) {
                    match bm {
                        ty::BindByValue(hir::MutMutable) => {}
                        _ => return,
                    }

                    mutables.entry(ident.name).or_default().push((hir_id, span));
                } else {
                    tcx.sess.delay_span_bug(span, "missing binding mode");
                }
            });
        }

        for (_name, ids) in mutables {
            // If any id for this name was used mutably then consider them all
            // ok, so move on to the next
            if ids.iter().any(|&(ref hir_id, _)| self.used_mut.contains(hir_id)) {
                continue;
            }

            let (hir_id, span) = ids[0];
            if span.compiler_desugaring_kind().is_some() {
                // If the `mut` arises as part of a desugaring, we should ignore it.
                continue;
            }

            // Ok, every name wasn't used mutably, so issue a warning that this
            // didn't need to be mutable.
            let mut_span = tcx.sess.source_map().span_until_non_whitespace(span);
            tcx.struct_span_lint_hir(UNUSED_MUT,
                                     hir_id,
                                     span,
                                     "variable does not need to be mutable")
                .span_suggestion_short_with_applicability(
                    mut_span,
                    "remove this `mut`",
                    String::new(),
                    Applicability::MachineApplicable)
                .emit();
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for UnusedMutCx<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.bccx.tcx.hir())
    }

    fn visit_arm(&mut self, arm: &hir::Arm) {
        self.check_unused_mut_pat(&arm.pats)
    }

    fn visit_local(&mut self, local: &hir::Local) {
        self.check_unused_mut_pat(slice::from_ref(&local.pat));
    }
}

impl<'a, 'tcx> Visitor<'tcx> for UsedMutFinder<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.bccx.tcx.hir())
    }

    fn visit_nested_body(&mut self, id: hir::BodyId) {
        let def_id = self.bccx.tcx.hir().body_owner_def_id(id);
        self.set.extend(self.bccx.tcx.borrowck(def_id).used_mut_nodes.iter().cloned());
        self.visit_body(self.bccx.tcx.hir().body(id));
    }
}
