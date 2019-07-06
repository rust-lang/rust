//! This calculates the types which has storage which lives across a suspension point in a
//! generator from the perspective of typeck. The actual types used at runtime
//! is calculated in `rustc_mir::transform::generator` and may be a subset of the
//! types computed here.

use rustc::hir::def_id::DefId;
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::hir::{self, Pat, PatKind, Expr};
use rustc::middle::region::{self, YieldData};
use rustc::ty::{self, Ty};
use syntax_pos::Span;
use super::FnCtxt;
use crate::util::nodemap::FxHashMap;

struct InteriorVisitor<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    types: FxHashMap<Ty<'tcx>, usize>,
    region_scope_tree: &'tcx region::ScopeTree,
    expr_count: usize,
    kind: hir::GeneratorKind,
}

impl<'a, 'tcx> InteriorVisitor<'a, 'tcx> {
    fn record(&mut self,
              ty: Ty<'tcx>,
              scope: Option<region::Scope>,
              expr: Option<&'tcx Expr>,
              source_span: Span) {
        use syntax_pos::DUMMY_SP;

        debug!("generator_interior: attempting to record type {:?} {:?} {:?} {:?}",
               ty, scope, expr, source_span);


        let live_across_yield = scope.map(|s| {
            self.region_scope_tree.yield_in_scope(s).and_then(|yield_data| {
                // If we are recording an expression that is the last yield
                // in the scope, or that has a postorder CFG index larger
                // than the one of all of the yields, then its value can't
                // be storage-live (and therefore live) at any of the yields.
                //
                // See the mega-comment at `yield_in_scope` for a proof.

                debug!("comparing counts yield: {} self: {}, source_span = {:?}",
                       yield_data.expr_and_pat_count, self.expr_count, source_span);

                if yield_data.expr_and_pat_count >= self.expr_count {
                    Some(yield_data)
                } else {
                    None
                }
            })
        }).unwrap_or_else(|| Some(YieldData {
            span: DUMMY_SP,
            expr_and_pat_count: 0,
            source: match self.kind { // Guess based on the kind of the current generator.
                hir::GeneratorKind::Gen => hir::YieldSource::Yield,
                hir::GeneratorKind::Async => hir::YieldSource::Await,
            },
        }));

        if let Some(yield_data) = live_across_yield {
            let ty = self.fcx.resolve_vars_if_possible(&ty);

            debug!("type in expr = {:?}, scope = {:?}, type = {:?}, count = {}, yield_span = {:?}",
                   expr, scope, ty, self.expr_count, yield_data.span);

            if let Some((unresolved_type, unresolved_type_span)) =
                self.fcx.unresolved_type_vars(&ty)
            {
                let note = format!("the type is part of the {} because of this {}",
                                   self.kind,
                                   yield_data.source);

                // If unresolved type isn't a ty_var then unresolved_type_span is None
                self.fcx.need_type_info_err_in_generator(
                    self.kind,
                    unresolved_type_span.unwrap_or(source_span),
                    unresolved_type,
                )
                    .span_note(yield_data.span, &*note)
                    .emit();
            } else {
                // Map the type to the number of types added before it
                let entries = self.types.len();
                self.types.entry(&ty).or_insert(entries);
            }
        } else {
            debug!("no type in expr = {:?}, count = {:?}, span = {:?}",
                   expr, self.expr_count, expr.map(|e| e.span));
        }
    }
}

pub fn resolve_interior<'a, 'tcx>(
    fcx: &'a FnCtxt<'a, 'tcx>,
    def_id: DefId,
    body_id: hir::BodyId,
    interior: Ty<'tcx>,
    kind: hir::GeneratorKind,
) {
    let body = fcx.tcx.hir().body(body_id);
    let mut visitor = InteriorVisitor {
        fcx,
        types: FxHashMap::default(),
        region_scope_tree: fcx.tcx.region_scope_tree(def_id),
        expr_count: 0,
        kind,
    };
    intravisit::walk_body(&mut visitor, body);

    // Check that we visited the same amount of expressions and the RegionResolutionVisitor
    let region_expr_count = visitor.region_scope_tree.body_expr_count(body_id).unwrap();
    assert_eq!(region_expr_count, visitor.expr_count);

    let mut types: Vec<_> = visitor.types.drain().collect();

    // Sort types by insertion order
    types.sort_by_key(|t| t.1);

    // Extract type components
    let type_list = fcx.tcx.mk_type_list(types.into_iter().map(|t| t.0));

    // The types in the generator interior contain lifetimes local to the generator itself,
    // which should not be exposed outside of the generator. Therefore, we replace these
    // lifetimes with existentially-bound lifetimes, which reflect the exact value of the
    // lifetimes not being known by users.
    //
    // These lifetimes are used in auto trait impl checking (for example,
    // if a Sync generator contains an &'α T, we need to check whether &'α T: Sync),
    // so knowledge of the exact relationships between them isn't particularly important.

    debug!("Types in generator {:?}, span = {:?}", type_list, body.value.span);

    // Replace all regions inside the generator interior with late bound regions
    // Note that each region slot in the types gets a new fresh late bound region,
    // which means that none of the regions inside relate to any other, even if
    // typeck had previously found constraints that would cause them to be related.
    let mut counter = 0;
    let type_list = fcx.tcx.fold_regions(&type_list, &mut false, |_, current_depth| {
        counter += 1;
        fcx.tcx.mk_region(ty::ReLateBound(current_depth, ty::BrAnon(counter)))
    });

    let witness = fcx.tcx.mk_generator_witness(ty::Binder::bind(type_list));

    debug!("Types in generator after region replacement {:?}, span = {:?}",
            witness, body.value.span);

    // Unify the type variable inside the generator with the new witness
    match fcx.at(&fcx.misc(body.value.span), fcx.param_env).eq(interior, witness) {
        Ok(ok) => fcx.register_infer_ok_obligations(ok),
        _ => bug!(),
    }
}

// This visitor has to have the same visit_expr calls as RegionResolutionVisitor in
// librustc/middle/region.rs since `expr_count` is compared against the results
// there.
impl<'a, 'tcx> Visitor<'tcx> for InteriorVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_pat(&mut self, pat: &'tcx Pat) {
        intravisit::walk_pat(self, pat);

        self.expr_count += 1;

        if let PatKind::Binding(..) = pat.node {
            let scope = self.region_scope_tree.var_scope(pat.hir_id.local_id);
            let ty = self.fcx.tables.borrow().pat_ty(pat);
            self.record(ty, Some(scope), None, pat.span);
        }
    }

    fn visit_expr(&mut self, expr: &'tcx Expr) {
        intravisit::walk_expr(self, expr);

        self.expr_count += 1;

        let scope = self.region_scope_tree.temporary_scope(expr.hir_id.local_id);

        // Record the unadjusted type
        let ty = self.fcx.tables.borrow().expr_ty(expr);
        self.record(ty, scope, Some(expr), expr.span);

        // Also include the adjusted types, since these can result in MIR locals
        for adjustment in self.fcx.tables.borrow().expr_adjustments(expr) {
            self.record(adjustment.target, scope, Some(expr), expr.span);
        }
    }
}
