//! This calculates the types which has storage which lives across a suspension point in a
//! generator from the perspective of typeck. The actual types used at runtime
//! is calculated in `rustc_mir::transform::generator` and may be a subset of the
//! types computed here.

use super::FnCtxt;
use rustc_data_structures::fx::{FxHashSet, FxIndexSet};
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::hir_id::HirIdSet;
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::{Arm, Expr, ExprKind, Guard, HirId, Pat, PatKind};
use rustc_middle::middle::region::{self, YieldData};
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;
use smallvec::SmallVec;

struct InteriorVisitor<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    types: FxIndexSet<ty::GeneratorInteriorTypeCause<'tcx>>,
    region_scope_tree: &'tcx region::ScopeTree,
    expr_count: usize,
    kind: hir::GeneratorKind,
    prev_unresolved_span: Option<Span>,
    /// Match arm guards have temporary borrows from the pattern bindings.
    /// In case there is a yield point in a guard with a reference to such bindings,
    /// such borrows can span across this yield point.
    /// As such, we need to track these borrows and record them despite of the fact
    /// that they may succeed the said yield point in the post-order.
    guard_bindings: SmallVec<[SmallVec<[HirId; 4]>; 1]>,
    guard_bindings_set: HirIdSet,
}

impl<'a, 'tcx> InteriorVisitor<'a, 'tcx> {
    fn record(
        &mut self,
        ty: Ty<'tcx>,
        scope: Option<region::Scope>,
        expr: Option<&'tcx Expr<'tcx>>,
        source_span: Span,
        guard_borrowing_from_pattern: bool,
    ) {
        use rustc_span::DUMMY_SP;

        debug!(
            "generator_interior: attempting to record type {:?} {:?} {:?} {:?}",
            ty, scope, expr, source_span
        );

        let live_across_yield = scope
            .map(|s| {
                self.region_scope_tree.yield_in_scope(s).and_then(|yield_data| {
                    // If we are recording an expression that is the last yield
                    // in the scope, or that has a postorder CFG index larger
                    // than the one of all of the yields, then its value can't
                    // be storage-live (and therefore live) at any of the yields.
                    //
                    // See the mega-comment at `yield_in_scope` for a proof.

                    debug!(
                        "comparing counts yield: {} self: {}, source_span = {:?}",
                        yield_data.expr_and_pat_count, self.expr_count, source_span
                    );

                    // If it is a borrowing happening in the guard,
                    // it needs to be recorded regardless because they
                    // do live across this yield point.
                    if guard_borrowing_from_pattern
                        || yield_data.expr_and_pat_count >= self.expr_count
                    {
                        Some(yield_data)
                    } else {
                        None
                    }
                })
            })
            .unwrap_or_else(|| {
                Some(YieldData { span: DUMMY_SP, expr_and_pat_count: 0, source: self.kind.into() })
            });

        if let Some(yield_data) = live_across_yield {
            let ty = self.fcx.resolve_vars_if_possible(ty);
            debug!(
                "type in expr = {:?}, scope = {:?}, type = {:?}, count = {}, yield_span = {:?}",
                expr, scope, ty, self.expr_count, yield_data.span
            );

            if let Some((unresolved_type, unresolved_type_span)) =
                self.fcx.unresolved_type_vars(&ty)
            {
                let note = format!(
                    "the type is part of the {} because of this {}",
                    self.kind, yield_data.source
                );

                // If unresolved type isn't a ty_var then unresolved_type_span is None
                let span = self
                    .prev_unresolved_span
                    .unwrap_or_else(|| unresolved_type_span.unwrap_or(source_span));
                self.fcx
                    .need_type_info_err_in_generator(self.kind, span, unresolved_type)
                    .span_note(yield_data.span, &*note)
                    .emit();
            } else {
                // Insert the type into the ordered set.
                let scope_span = scope.map(|s| s.span(self.fcx.tcx, self.region_scope_tree));
                self.types.insert(ty::GeneratorInteriorTypeCause {
                    span: source_span,
                    ty: &ty,
                    scope_span,
                    yield_span: yield_data.span,
                    expr: expr.map(|e| e.hir_id),
                });
            }
        } else {
            debug!(
                "no type in expr = {:?}, count = {:?}, span = {:?}",
                expr,
                self.expr_count,
                expr.map(|e| e.span)
            );
            let ty = self.fcx.resolve_vars_if_possible(ty);
            if let Some((unresolved_type, unresolved_type_span)) =
                self.fcx.unresolved_type_vars(&ty)
            {
                debug!(
                    "remained unresolved_type = {:?}, unresolved_type_span: {:?}",
                    unresolved_type, unresolved_type_span
                );
                self.prev_unresolved_span = unresolved_type_span;
            }
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
        types: FxIndexSet::default(),
        region_scope_tree: fcx.tcx.region_scope_tree(def_id),
        expr_count: 0,
        kind,
        prev_unresolved_span: None,
        guard_bindings: <_>::default(),
        guard_bindings_set: <_>::default(),
    };
    intravisit::walk_body(&mut visitor, body);

    // Check that we visited the same amount of expressions and the RegionResolutionVisitor
    let region_expr_count = visitor.region_scope_tree.body_expr_count(body_id).unwrap();
    assert_eq!(region_expr_count, visitor.expr_count);

    // The types are already kept in insertion order.
    let types = visitor.types;

    // The types in the generator interior contain lifetimes local to the generator itself,
    // which should not be exposed outside of the generator. Therefore, we replace these
    // lifetimes with existentially-bound lifetimes, which reflect the exact value of the
    // lifetimes not being known by users.
    //
    // These lifetimes are used in auto trait impl checking (for example,
    // if a Sync generator contains an &'α T, we need to check whether &'α T: Sync),
    // so knowledge of the exact relationships between them isn't particularly important.

    debug!("types in generator {:?}, span = {:?}", types, body.value.span);

    let mut counter = 0;
    let mut captured_tys = FxHashSet::default();
    let type_causes: Vec<_> = types
        .into_iter()
        .filter_map(|mut cause| {
            // Erase regions and canonicalize late-bound regions to deduplicate as many types as we
            // can.
            let erased = fcx.tcx.erase_regions(cause.ty);
            if captured_tys.insert(erased) {
                // Replace all regions inside the generator interior with late bound regions.
                // Note that each region slot in the types gets a new fresh late bound region,
                // which means that none of the regions inside relate to any other, even if
                // typeck had previously found constraints that would cause them to be related.
                let folded = fcx.tcx.fold_regions(erased, &mut false, |_, current_depth| {
                    let r = fcx.tcx.mk_region(ty::ReLateBound(current_depth, ty::BrAnon(counter)));
                    counter += 1;
                    r
                });

                cause.ty = folded;
                Some(cause)
            } else {
                None
            }
        })
        .collect();

    // Extract type components to build the witness type.
    let type_list = fcx.tcx.mk_type_list(type_causes.iter().map(|cause| cause.ty));
    let witness = fcx.tcx.mk_generator_witness(ty::Binder::bind(type_list));

    // Store the generator types and spans into the typeck results for this generator.
    visitor.fcx.inh.typeck_results.borrow_mut().generator_interior_types = type_causes;

    debug!(
        "types in generator after region replacement {:?}, span = {:?}",
        witness, body.value.span
    );

    // Unify the type variable inside the generator with the new witness
    match fcx.at(&fcx.misc(body.value.span), fcx.param_env).eq(interior, witness) {
        Ok(ok) => fcx.register_infer_ok_obligations(ok),
        _ => bug!(),
    }
}

// This visitor has to have the same visit_expr calls as RegionResolutionVisitor in
// librustc_middle/middle/region.rs since `expr_count` is compared against the results
// there.
impl<'a, 'tcx> Visitor<'tcx> for InteriorVisitor<'a, 'tcx> {
    type Map = intravisit::ErasedMap<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }

    fn visit_arm(&mut self, arm: &'tcx Arm<'tcx>) {
        let Arm { guard, pat, body, .. } = arm;
        self.visit_pat(pat);
        if let Some(ref g) = guard {
            self.guard_bindings.push(<_>::default());
            ArmPatCollector {
                guard_bindings_set: &mut self.guard_bindings_set,
                guard_bindings: self
                    .guard_bindings
                    .last_mut()
                    .expect("should have pushed at least one earlier"),
            }
            .visit_pat(pat);

            match g {
                Guard::If(ref e) => {
                    self.visit_expr(e);
                }
                Guard::IfLet(ref pat, ref e) => {
                    self.visit_pat(pat);
                    self.visit_expr(e);
                }
            }

            let mut scope_var_ids =
                self.guard_bindings.pop().expect("should have pushed at least one earlier");
            for var_id in scope_var_ids.drain(..) {
                self.guard_bindings_set.remove(&var_id);
            }
        }
        self.visit_expr(body);
    }

    fn visit_pat(&mut self, pat: &'tcx Pat<'tcx>) {
        intravisit::walk_pat(self, pat);

        self.expr_count += 1;

        if let PatKind::Binding(..) = pat.kind {
            let scope = self.region_scope_tree.var_scope(pat.hir_id.local_id);
            let ty = self.fcx.typeck_results.borrow().pat_ty(pat);
            self.record(ty, Some(scope), None, pat.span, false);
        }
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        let mut guard_borrowing_from_pattern = false;
        match &expr.kind {
            ExprKind::Call(callee, args) => match &callee.kind {
                ExprKind::Path(qpath) => {
                    let res = self.fcx.typeck_results.borrow().qpath_res(qpath, callee.hir_id);
                    match res {
                        // Direct calls never need to keep the callee `ty::FnDef`
                        // ZST in a temporary, so skip its type, just in case it
                        // can significantly complicate the generator type.
                        Res::Def(
                            DefKind::Fn | DefKind::AssocFn | DefKind::Ctor(_, CtorKind::Fn),
                            _,
                        ) => {
                            // NOTE(eddyb) this assumes a path expression has
                            // no nested expressions to keep track of.
                            self.expr_count += 1;

                            // Record the rest of the call expression normally.
                            for arg in *args {
                                self.visit_expr(arg);
                            }
                        }
                        _ => intravisit::walk_expr(self, expr),
                    }
                }
                _ => intravisit::walk_expr(self, expr),
            },
            ExprKind::Path(qpath) => {
                intravisit::walk_expr(self, expr);
                let res = self.fcx.typeck_results.borrow().qpath_res(qpath, expr.hir_id);
                match res {
                    Res::Local(id) if self.guard_bindings_set.contains(&id) => {
                        guard_borrowing_from_pattern = true;
                    }
                    _ => {}
                }
            }
            _ => intravisit::walk_expr(self, expr),
        }

        self.expr_count += 1;

        let scope = self.region_scope_tree.temporary_scope(expr.hir_id.local_id);

        // If there are adjustments, then record the final type --
        // this is the actual value that is being produced.
        if let Some(adjusted_ty) = self.fcx.typeck_results.borrow().expr_ty_adjusted_opt(expr) {
            self.record(adjusted_ty, scope, Some(expr), expr.span, guard_borrowing_from_pattern);
        }

        // Also record the unadjusted type (which is the only type if
        // there are no adjustments). The reason for this is that the
        // unadjusted value is sometimes a "temporary" that would wind
        // up in a MIR temporary.
        //
        // As an example, consider an expression like `vec![].push(x)`.
        // Here, the `vec![]` would wind up MIR stored into a
        // temporary variable `t` which we can borrow to invoke
        // `<Vec<_>>::push(&mut t, x)`.
        //
        // Note that an expression can have many adjustments, and we
        // are just ignoring those intermediate types. This is because
        // those intermediate values are always linearly "consumed" by
        // the other adjustments, and hence would never be directly
        // captured in the MIR.
        //
        // (Note that this partly relies on the fact that the `Deref`
        // traits always return references, which means their content
        // can be reborrowed without needing to spill to a temporary.
        // If this were not the case, then we could conceivably have
        // to create intermediate temporaries.)
        //
        // The type table might not have information for this expression
        // if it is in a malformed scope. (#66387)
        if let Some(ty) = self.fcx.typeck_results.borrow().expr_ty_opt(expr) {
            if guard_borrowing_from_pattern {
                // Match guards create references to all the bindings in the pattern that are used
                // in the guard, e.g. `y if is_even(y) => ...` becomes `is_even(*r_y)` where `r_y`
                // is a reference to `y`, so we must record a reference to the type of the binding.
                let tcx = self.fcx.tcx;
                let ref_ty = tcx.mk_ref(
                    // Use `ReErased` as `resolve_interior` is going to replace all the regions anyway.
                    tcx.mk_region(ty::RegionKind::ReErased),
                    ty::TypeAndMut { ty, mutbl: hir::Mutability::Not },
                );
                self.record(ref_ty, scope, Some(expr), expr.span, guard_borrowing_from_pattern);
            }
            self.record(ty, scope, Some(expr), expr.span, guard_borrowing_from_pattern);
        } else {
            self.fcx.tcx.sess.delay_span_bug(expr.span, "no type for node");
        }
    }
}

struct ArmPatCollector<'a> {
    guard_bindings_set: &'a mut HirIdSet,
    guard_bindings: &'a mut SmallVec<[HirId; 4]>,
}

impl<'a, 'tcx> Visitor<'tcx> for ArmPatCollector<'a> {
    type Map = intravisit::ErasedMap<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }

    fn visit_pat(&mut self, pat: &'tcx Pat<'tcx>) {
        intravisit::walk_pat(self, pat);
        if let PatKind::Binding(_, id, ..) = pat.kind {
            self.guard_bindings.push(id);
            self.guard_bindings_set.insert(id);
        }
    }
}
