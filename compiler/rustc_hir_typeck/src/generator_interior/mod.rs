//! This calculates the types which has storage which lives across a suspension point in a
//! generator from the perspective of typeck. The actual types used at runtime
//! is calculated in `rustc_mir_transform::generator` and may be a subset of the
//! types computed here.

use self::drop_ranges::DropRanges;
use super::FnCtxt;
use rustc_data_structures::fx::{FxHashSet, FxIndexSet};
use rustc_errors::{pluralize, DelayDm};
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::hir_id::HirIdSet;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{Arm, Expr, ExprKind, Guard, HirId, Pat, PatKind};
use rustc_infer::infer::{DefineOpaqueTypes, RegionVariableOrigin};
use rustc_middle::middle::region::{self, Scope, ScopeData, YieldData};
use rustc_middle::ty::fold::FnMutDelegate;
use rustc_middle::ty::{self, BoundVariableKind, RvalueScopes, Ty, TyCtxt, TypeVisitableExt};
use rustc_span::symbol::sym;
use rustc_span::Span;
use smallvec::{smallvec, SmallVec};

mod drop_ranges;

struct InteriorVisitor<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    region_scope_tree: &'a region::ScopeTree,
    types: FxIndexSet<ty::GeneratorInteriorTypeCause<'tcx>>,
    rvalue_scopes: &'a RvalueScopes,
    expr_count: usize,
    kind: hir::GeneratorKind,
    prev_unresolved_span: Option<Span>,
    linted_values: HirIdSet,
    drop_ranges: DropRanges,
}

impl<'a, 'tcx> InteriorVisitor<'a, 'tcx> {
    fn record(
        &mut self,
        ty: Ty<'tcx>,
        hir_id: HirId,
        scope: Option<region::Scope>,
        expr: Option<&'tcx Expr<'tcx>>,
        source_span: Span,
    ) {
        use rustc_span::DUMMY_SP;

        let ty = self.fcx.resolve_vars_if_possible(ty);

        debug!(
            "attempting to record type ty={:?}; hir_id={:?}; scope={:?}; expr={:?}; source_span={:?}; expr_count={:?}",
            ty, hir_id, scope, expr, source_span, self.expr_count,
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

                    yield_data
                        .iter()
                        .find(|yield_data| {
                            debug!(
                                "comparing counts yield: {} self: {}, source_span = {:?}",
                                yield_data.expr_and_pat_count, self.expr_count, source_span
                            );

                            if self
                                .is_dropped_at_yield_location(hir_id, yield_data.expr_and_pat_count)
                            {
                                debug!("value is dropped at yield point; not recording");
                                return false;
                            }

                            // If it is a borrowing happening in the guard,
                            // it needs to be recorded regardless because they
                            // do live across this yield point.
                            yield_data.expr_and_pat_count >= self.expr_count
                        })
                        .cloned()
                })
            })
            .unwrap_or_else(|| {
                Some(YieldData { span: DUMMY_SP, expr_and_pat_count: 0, source: self.kind.into() })
            });

        if let Some(yield_data) = live_across_yield {
            debug!(
                "type in expr = {:?}, scope = {:?}, type = {:?}, count = {}, yield_span = {:?}",
                expr, scope, ty, self.expr_count, yield_data.span
            );

            if let Some((unresolved_term, unresolved_type_span)) =
                self.fcx.first_unresolved_const_or_ty_var(&ty)
            {
                // If unresolved type isn't a ty_var then unresolved_type_span is None
                let span = self
                    .prev_unresolved_span
                    .unwrap_or_else(|| unresolved_type_span.unwrap_or(source_span));

                // If we encounter an int/float variable, then inference fallback didn't
                // finish due to some other error. Don't emit spurious additional errors.
                if let Some(unresolved_ty) = unresolved_term.ty()
                    && let ty::Infer(ty::InferTy::IntVar(_) | ty::InferTy::FloatVar(_)) = unresolved_ty.kind()
                {
                    self.fcx
                        .tcx
                        .sess
                        .delay_span_bug(span, &format!("Encountered var {:?}", unresolved_term));
                } else {
                    let note = format!(
                        "the type is part of the {} because of this {}",
                        self.kind.descr(),
                        yield_data.source
                    );

                    self.fcx
                        .need_type_info_err_in_generator(self.kind, span, unresolved_term)
                        .span_note(yield_data.span, &*note)
                        .emit();
                }
            } else {
                // Insert the type into the ordered set.
                let scope_span = scope.map(|s| s.span(self.fcx.tcx, self.region_scope_tree));

                if !self.linted_values.contains(&hir_id) {
                    check_must_not_suspend_ty(
                        self.fcx,
                        ty,
                        hir_id,
                        SuspendCheckData {
                            expr,
                            source_span,
                            yield_span: yield_data.span,
                            plural_len: 1,
                            ..Default::default()
                        },
                    );
                    self.linted_values.insert(hir_id);
                }

                self.types.insert(ty::GeneratorInteriorTypeCause {
                    span: source_span,
                    ty,
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
            if let Some((unresolved_type, unresolved_type_span)) =
                self.fcx.first_unresolved_const_or_ty_var(&ty)
            {
                debug!(
                    "remained unresolved_type = {:?}, unresolved_type_span: {:?}",
                    unresolved_type, unresolved_type_span
                );
                self.prev_unresolved_span = unresolved_type_span;
            }
        }
    }

    /// If drop tracking is enabled, consult drop_ranges to see if a value is
    /// known to be dropped at a yield point and therefore can be omitted from
    /// the generator witness.
    fn is_dropped_at_yield_location(&self, value_hir_id: HirId, yield_location: usize) -> bool {
        // short-circuit if drop tracking is not enabled.
        if !self.fcx.sess().opts.unstable_opts.drop_tracking {
            return false;
        }

        self.drop_ranges.is_dropped_at(value_hir_id, yield_location)
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
    let typeck_results = fcx.inh.typeck_results.borrow();
    let mut visitor = InteriorVisitor {
        fcx,
        types: FxIndexSet::default(),
        region_scope_tree: fcx.tcx.region_scope_tree(def_id),
        rvalue_scopes: &typeck_results.rvalue_scopes,
        expr_count: 0,
        kind,
        prev_unresolved_span: None,
        linted_values: <_>::default(),
        drop_ranges: drop_ranges::compute_drop_ranges(fcx, def_id, body),
    };
    intravisit::walk_body(&mut visitor, body);

    // Check that we visited the same amount of expressions as the RegionResolutionVisitor
    let region_expr_count = fcx.tcx.region_scope_tree(def_id).body_expr_count(body_id).unwrap();
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

    // We want to deduplicate if the lifetimes are the same modulo some non-informative counter.
    // So, we need to actually do two passes: first by type to anonymize (preserving information
    // required for diagnostics), then a second pass over all captured types to reassign disjoint
    // region indices.
    let mut captured_tys = FxHashSet::default();
    let type_causes: Vec<_> = types
        .into_iter()
        .filter_map(|mut cause| {
            // Replace all regions inside the generator interior with late bound regions.
            // Note that each region slot in the types gets a new fresh late bound region,
            // which means that none of the regions inside relate to any other, even if
            // typeck had previously found constraints that would cause them to be related.

            let mut counter = 0;
            let mut mk_bound_region = |span| {
                let kind = ty::BrAnon(counter, span);
                let var = ty::BoundVar::from_u32(counter);
                counter += 1;
                ty::BoundRegion { var, kind }
            };
            let ty = fcx.normalize(cause.span, cause.ty);
            let ty = fcx.tcx.fold_regions(ty, |region, current_depth| {
                let br = match region.kind() {
                    ty::ReVar(vid) => {
                        let origin = fcx.region_var_origin(vid);
                        match origin {
                            RegionVariableOrigin::EarlyBoundRegion(span, _) => {
                                mk_bound_region(Some(span))
                            }
                            _ => mk_bound_region(None),
                        }
                    }
                    // FIXME: these should use `BrNamed`
                    ty::ReEarlyBound(region) => {
                        mk_bound_region(Some(fcx.tcx.def_span(region.def_id)))
                    }
                    ty::ReLateBound(_, ty::BoundRegion { kind, .. })
                    | ty::ReFree(ty::FreeRegion { bound_region: kind, .. }) => match kind {
                        ty::BoundRegionKind::BrAnon(_, span) => mk_bound_region(span),
                        ty::BoundRegionKind::BrNamed(def_id, _) => {
                            mk_bound_region(Some(fcx.tcx.def_span(def_id)))
                        }
                        ty::BoundRegionKind::BrEnv => mk_bound_region(None),
                    },
                    _ => mk_bound_region(None),
                };
                let r = fcx.tcx.mk_re_late_bound(current_depth, br);
                r
            });
            captured_tys.insert(ty).then(|| {
                cause.ty = ty;
                cause
            })
        })
        .collect();

    let mut bound_vars: SmallVec<[BoundVariableKind; 4]> = smallvec![];
    let mut counter = 0;
    // Optimization: If there is only one captured type, then we don't actually
    // need to fold and reindex (since the first type doesn't change).
    let type_causes = if captured_tys.len() > 0 {
        // Optimization: Use `replace_escaping_bound_vars_uncached` instead of
        // `fold_regions`, since we only have late bound regions, and it skips
        // types without bound regions.
        fcx.tcx.replace_escaping_bound_vars_uncached(
            type_causes,
            FnMutDelegate {
                regions: &mut |br| {
                    let kind = match br.kind {
                        ty::BrAnon(_, span) => ty::BrAnon(counter, span),
                        _ => br.kind,
                    };
                    let var = ty::BoundVar::from_usize(bound_vars.len());
                    bound_vars.push(ty::BoundVariableKind::Region(kind));
                    counter += 1;
                    fcx.tcx.mk_re_late_bound(ty::INNERMOST, ty::BoundRegion { var, kind })
                },
                types: &mut |b| bug!("unexpected bound ty in binder: {b:?}"),
                consts: &mut |b, ty| bug!("unexpected bound ct in binder: {b:?} {ty}"),
            },
        )
    } else {
        type_causes
    };

    // Extract type components to build the witness type.
    let type_list = fcx.tcx.mk_type_list_from_iter(type_causes.iter().map(|cause| cause.ty));
    let bound_vars = fcx.tcx.mk_bound_variable_kinds(&bound_vars);
    let witness =
        fcx.tcx.mk_generator_witness(ty::Binder::bind_with_vars(type_list, bound_vars.clone()));

    drop(typeck_results);
    // Store the generator types and spans into the typeck results for this generator.
    fcx.inh.typeck_results.borrow_mut().generator_interior_types =
        ty::Binder::bind_with_vars(type_causes, bound_vars);

    debug!(
        "types in generator after region replacement {:?}, span = {:?}",
        witness, body.value.span
    );

    // Unify the type variable inside the generator with the new witness
    match fcx.at(&fcx.misc(body.value.span), fcx.param_env).eq(
        DefineOpaqueTypes::No,
        interior,
        witness,
    ) {
        Ok(ok) => fcx.register_infer_ok_obligations(ok),
        _ => bug!("failed to relate {interior} and {witness}"),
    }
}

// This visitor has to have the same visit_expr calls as RegionResolutionVisitor in
// librustc_middle/middle/region.rs since `expr_count` is compared against the results
// there.
impl<'a, 'tcx> Visitor<'tcx> for InteriorVisitor<'a, 'tcx> {
    fn visit_arm(&mut self, arm: &'tcx Arm<'tcx>) {
        let Arm { guard, pat, body, .. } = arm;
        self.visit_pat(pat);
        if let Some(ref g) = guard {
            {
                // If there is a guard, we need to count all variables bound in the pattern as
                // borrowed for the entire guard body, regardless of whether they are accessed.
                // We do this by walking the pattern bindings and recording `&T` for any `x: T`
                // that is bound.

                struct ArmPatCollector<'a, 'b, 'tcx> {
                    interior_visitor: &'a mut InteriorVisitor<'b, 'tcx>,
                    scope: Scope,
                }

                impl<'a, 'b, 'tcx> Visitor<'tcx> for ArmPatCollector<'a, 'b, 'tcx> {
                    fn visit_pat(&mut self, pat: &'tcx Pat<'tcx>) {
                        intravisit::walk_pat(self, pat);
                        if let PatKind::Binding(_, id, ident, ..) = pat.kind {
                            let ty =
                                self.interior_visitor.fcx.typeck_results.borrow().node_type(id);
                            let tcx = self.interior_visitor.fcx.tcx;
                            let ty = tcx.mk_ref(
                                // Use `ReErased` as `resolve_interior` is going to replace all the
                                // regions anyway.
                                tcx.lifetimes.re_erased,
                                ty::TypeAndMut { ty, mutbl: hir::Mutability::Not },
                            );
                            self.interior_visitor.record(
                                ty,
                                id,
                                Some(self.scope),
                                None,
                                ident.span,
                            );
                        }
                    }
                }

                ArmPatCollector {
                    interior_visitor: self,
                    scope: Scope { id: g.body().hir_id.local_id, data: ScopeData::Node },
                }
                .visit_pat(pat);
            }

            match g {
                Guard::If(ref e) => {
                    self.visit_expr(e);
                }
                Guard::IfLet(ref l) => {
                    self.visit_let_expr(l);
                }
            }
        }
        self.visit_expr(body);
    }

    fn visit_pat(&mut self, pat: &'tcx Pat<'tcx>) {
        intravisit::walk_pat(self, pat);

        self.expr_count += 1;

        if let PatKind::Binding(..) = pat.kind {
            let scope = self.region_scope_tree.var_scope(pat.hir_id.local_id).unwrap();
            let ty = self.fcx.typeck_results.borrow().pat_ty(pat);
            self.record(ty, pat.hir_id, Some(scope), None, pat.span);
        }
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
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
            _ => intravisit::walk_expr(self, expr),
        }

        self.expr_count += 1;

        debug!("is_borrowed_temporary: {:?}", self.drop_ranges.is_borrowed_temporary(expr));

        let ty = self.fcx.typeck_results.borrow().expr_ty_adjusted_opt(expr);

        // Typically, the value produced by an expression is consumed by its parent in some way,
        // so we only have to check if the parent contains a yield (note that the parent may, for
        // example, store the value into a local variable, but then we already consider local
        // variables to be live across their scope).
        //
        // However, in the case of temporary values, we are going to store the value into a
        // temporary on the stack that is live for the current temporary scope and then return a
        // reference to it. That value may be live across the entire temporary scope.
        //
        // There's another subtlety: if the type has an observable drop, it must be dropped after
        // the yield, even if it's not borrowed or referenced after the yield. Ideally this would
        // *only* happen for types with observable drop, not all types which wrap them, but that
        // doesn't match the behavior of MIR borrowck and causes ICEs. See the FIXME comment in
        // tests/ui/generator/drop-tracking-parent-expression.rs.
        let scope = if self.drop_ranges.is_borrowed_temporary(expr)
            || ty.map_or(true, |ty| {
                // Avoid ICEs in needs_drop.
                let ty = self.fcx.resolve_vars_if_possible(ty);
                let ty = self.fcx.tcx.erase_regions(ty);
                if ty.needs_infer() {
                    self.fcx
                        .tcx
                        .sess
                        .delay_span_bug(expr.span, &format!("inference variables in {ty}"));
                    true
                } else {
                    ty.needs_drop(self.fcx.tcx, self.fcx.param_env)
                }
            }) {
            self.rvalue_scopes.temporary_scope(self.region_scope_tree, expr.hir_id.local_id)
        } else {
            let parent_expr = self
                .fcx
                .tcx
                .hir()
                .parent_iter(expr.hir_id)
                .find(|(_, node)| matches!(node, hir::Node::Expr(_)))
                .map(|(id, _)| id);
            debug!("parent_expr: {:?}", parent_expr);
            match parent_expr {
                Some(parent) => Some(Scope { id: parent.local_id, data: ScopeData::Node }),
                None => {
                    self.rvalue_scopes.temporary_scope(self.region_scope_tree, expr.hir_id.local_id)
                }
            }
        };

        // If there are adjustments, then record the final type --
        // this is the actual value that is being produced.
        if let Some(adjusted_ty) = ty {
            self.record(adjusted_ty, expr.hir_id, scope, Some(expr), expr.span);
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
            self.record(ty, expr.hir_id, scope, Some(expr), expr.span);
        } else {
            self.fcx.tcx.sess.delay_span_bug(expr.span, "no type for node");
        }
    }
}

#[derive(Default)]
struct SuspendCheckData<'a, 'tcx> {
    expr: Option<&'tcx Expr<'tcx>>,
    source_span: Span,
    yield_span: Span,
    descr_pre: &'a str,
    descr_post: &'a str,
    plural_len: usize,
}

// Returns whether it emitted a diagnostic or not
// Note that this fn and the proceeding one are based on the code
// for creating must_use diagnostics
//
// Note that this technique was chosen over things like a `Suspend` marker trait
// as it is simpler and has precedent in the compiler
fn check_must_not_suspend_ty<'tcx>(
    fcx: &FnCtxt<'_, 'tcx>,
    ty: Ty<'tcx>,
    hir_id: HirId,
    data: SuspendCheckData<'_, 'tcx>,
) -> bool {
    if ty.is_unit()
    // FIXME: should this check `Ty::is_inhabited_from`. This query is not available in this stage
    // of typeck (before ReVar and RePlaceholder are removed), but may remove noise, like in
    // `must_use`
    // || !ty.is_inhabited_from(fcx.tcx, fcx.tcx.parent_module(hir_id).to_def_id(), fcx.param_env)
    {
        return false;
    }

    let plural_suffix = pluralize!(data.plural_len);

    debug!("Checking must_not_suspend for {}", ty);

    match *ty.kind() {
        ty::Adt(..) if ty.is_box() => {
            let boxed_ty = ty.boxed_ty();
            let descr_pre = &format!("{}boxed ", data.descr_pre);
            check_must_not_suspend_ty(fcx, boxed_ty, hir_id, SuspendCheckData { descr_pre, ..data })
        }
        ty::Adt(def, _) => check_must_not_suspend_def(fcx.tcx, def.did(), hir_id, data),
        // FIXME: support adding the attribute to TAITs
        ty::Alias(ty::Opaque, ty::AliasTy { def_id: def, .. }) => {
            let mut has_emitted = false;
            for &(predicate, _) in fcx.tcx.explicit_item_bounds(def) {
                // We only look at the `DefId`, so it is safe to skip the binder here.
                if let ty::PredicateKind::Clause(ty::Clause::Trait(ref poly_trait_predicate)) =
                    predicate.kind().skip_binder()
                {
                    let def_id = poly_trait_predicate.trait_ref.def_id;
                    let descr_pre = &format!("{}implementer{} of ", data.descr_pre, plural_suffix);
                    if check_must_not_suspend_def(
                        fcx.tcx,
                        def_id,
                        hir_id,
                        SuspendCheckData { descr_pre, ..data },
                    ) {
                        has_emitted = true;
                        break;
                    }
                }
            }
            has_emitted
        }
        ty::Dynamic(binder, _, _) => {
            let mut has_emitted = false;
            for predicate in binder.iter() {
                if let ty::ExistentialPredicate::Trait(ref trait_ref) = predicate.skip_binder() {
                    let def_id = trait_ref.def_id;
                    let descr_post = &format!(" trait object{}{}", plural_suffix, data.descr_post);
                    if check_must_not_suspend_def(
                        fcx.tcx,
                        def_id,
                        hir_id,
                        SuspendCheckData { descr_post, ..data },
                    ) {
                        has_emitted = true;
                        break;
                    }
                }
            }
            has_emitted
        }
        ty::Tuple(fields) => {
            let mut has_emitted = false;
            let comps = match data.expr.map(|e| &e.kind) {
                Some(hir::ExprKind::Tup(comps)) if comps.len() == fields.len() => Some(comps),
                _ => None,
            };
            for (i, ty) in fields.iter().enumerate() {
                let descr_post = &format!(" in tuple element {i}");
                let span = comps.and_then(|c| c.get(i)).map(|e| e.span).unwrap_or(data.source_span);
                if check_must_not_suspend_ty(
                    fcx,
                    ty,
                    hir_id,
                    SuspendCheckData {
                        descr_post,
                        expr: comps.and_then(|comps| comps.get(i)),
                        source_span: span,
                        ..data
                    },
                ) {
                    has_emitted = true;
                }
            }
            has_emitted
        }
        ty::Array(ty, len) => {
            let descr_pre = &format!("{}array{} of ", data.descr_pre, plural_suffix);
            check_must_not_suspend_ty(
                fcx,
                ty,
                hir_id,
                SuspendCheckData {
                    descr_pre,
                    plural_len: len.try_eval_target_usize(fcx.tcx, fcx.param_env).unwrap_or(0)
                        as usize
                        + 1,
                    ..data
                },
            )
        }
        // If drop tracking is enabled, we want to look through references, since the referrent
        // may not be considered live across the await point.
        ty::Ref(_region, ty, _mutability) if fcx.sess().opts.unstable_opts.drop_tracking => {
            let descr_pre = &format!("{}reference{} to ", data.descr_pre, plural_suffix);
            check_must_not_suspend_ty(fcx, ty, hir_id, SuspendCheckData { descr_pre, ..data })
        }
        _ => false,
    }
}

fn check_must_not_suspend_def(
    tcx: TyCtxt<'_>,
    def_id: DefId,
    hir_id: HirId,
    data: SuspendCheckData<'_, '_>,
) -> bool {
    if let Some(attr) = tcx.get_attr(def_id, sym::must_not_suspend) {
        tcx.struct_span_lint_hir(
            rustc_session::lint::builtin::MUST_NOT_SUSPEND,
            hir_id,
            data.source_span,
            DelayDm(|| {
                format!(
                    "{}`{}`{} held across a suspend point, but should not be",
                    data.descr_pre,
                    tcx.def_path_str(def_id),
                    data.descr_post,
                )
            }),
            |lint| {
                // add span pointing to the offending yield/await
                lint.span_label(data.yield_span, "the value is held across this suspend point");

                // Add optional reason note
                if let Some(note) = attr.value_str() {
                    // FIXME(guswynn): consider formatting this better
                    lint.span_note(data.source_span, note.as_str());
                }

                // Add some quick suggestions on what to do
                // FIXME: can `drop` work as a suggestion here as well?
                lint.span_help(
                    data.source_span,
                    "consider using a block (`{ ... }`) \
                    to shrink the value's scope, ending before the suspend point",
                );

                lint
            },
        );

        true
    } else {
        false
    }
}
