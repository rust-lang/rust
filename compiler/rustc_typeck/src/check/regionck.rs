//! The region check is a final pass that runs over the AST after we have
//! inferred the type constraints but before we have actually finalized
//! the types. Its purpose is to embed a variety of region constraints.
//! Inserting these constraints as a separate pass is good because (1) it
//! localizes the code that has to do with region inference and (2) often
//! we cannot know what constraints are needed until the basic types have
//! been inferred.
//!
//! ### Interaction with the borrow checker
//!
//! In general, the job of the borrowck module (which runs later) is to
//! check that all soundness criteria are met, given a particular set of
//! regions. The job of *this* module is to anticipate the needs of the
//! borrow checker and infer regions that will satisfy its requirements.
//! It is generally true that the inference doesn't need to be sound,
//! meaning that if there is a bug and we inferred bad regions, the borrow
//! checker should catch it. This is not entirely true though; for
//! example, the borrow checker doesn't check subtyping, and it doesn't
//! check that region pointers are always live when they are used. It
//! might be worthwhile to fix this so that borrowck serves as a kind of
//! verification step -- that would add confidence in the overall
//! correctness of the compiler, at the cost of duplicating some type
//! checks and effort.
//!
//! ### Inferring the duration of borrows, automatic and otherwise
//!
//! Whenever we introduce a borrowed pointer, for example as the result of
//! a borrow expression `let x = &data`, the lifetime of the pointer `x`
//! is always specified as a region inference variable. `regionck` has the
//! job of adding constraints such that this inference variable is as
//! narrow as possible while still accommodating all uses (that is, every
//! dereference of the resulting pointer must be within the lifetime).
//!
//! #### Reborrows
//!
//! Generally speaking, `regionck` does NOT try to ensure that the data
//! `data` will outlive the pointer `x`. That is the job of borrowck. The
//! one exception is when "re-borrowing" the contents of another borrowed
//! pointer. For example, imagine you have a borrowed pointer `b` with
//! lifetime `L1` and you have an expression `&*b`. The result of this
//! expression will be another borrowed pointer with lifetime `L2` (which is
//! an inference variable). The borrow checker is going to enforce the
//! constraint that `L2 < L1`, because otherwise you are re-borrowing data
//! for a lifetime larger than the original loan. However, without the
//! routines in this module, the region inferencer would not know of this
//! dependency and thus it might infer the lifetime of `L2` to be greater
//! than `L1` (issue #3148).
//!
//! There are a number of troublesome scenarios in the tests
//! `region-dependent-*.rs`, but here is one example:
//!
//!     struct Foo { i: i32 }
//!     struct Bar { foo: Foo  }
//!     fn get_i<'a>(x: &'a Bar) -> &'a i32 {
//!        let foo = &x.foo; // Lifetime L1
//!        &foo.i            // Lifetime L2
//!     }
//!
//! Note that this comes up either with `&` expressions, `ref`
//! bindings, and `autorefs`, which are the three ways to introduce
//! a borrow.
//!
//! The key point here is that when you are borrowing a value that
//! is "guaranteed" by a borrowed pointer, you must link the
//! lifetime of that borrowed pointer (`L1`, here) to the lifetime of
//! the borrow itself (`L2`). What do I mean by "guaranteed" by a
//! borrowed pointer? I mean any data that is reached by first
//! dereferencing a borrowed pointer and then either traversing
//! interior offsets or boxes. We say that the guarantor
//! of such data is the region of the borrowed pointer that was
//! traversed. This is essentially the same as the ownership
//! relation, except that a borrowed pointer never owns its
//! contents.

use crate::check::dropck;
use crate::check::FnCtxt;
use crate::mem_categorization as mc;
use crate::middle::region;
use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::PatKind;
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::{self, RegionObligation, RegionckMode};
use rustc_middle::hir::place::{PlaceBase, PlaceWithHirId};
use rustc_middle::ty::adjustment;
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;
use rustc_trait_selection::infer::OutlivesEnvironmentExt;
use rustc_trait_selection::opaque_types::InferCtxtExt;
use std::ops::Deref;

// a variation on try that just returns unit
macro_rules! ignore_err {
    ($e:expr) => {
        match $e {
            Ok(e) => e,
            Err(_) => {
                debug!("ignoring mem-categorization error!");
                return ();
            }
        }
    };
}

///////////////////////////////////////////////////////////////////////////
// PUBLIC ENTRY POINTS

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub fn regionck_expr(&self, body: &'tcx hir::Body<'tcx>) {
        let subject = self.tcx.hir().body_owner_def_id(body.id());
        let id = body.value.hir_id;
        let mut rcx = RegionCtxt::new(self, id, Subject(subject), self.param_env);

        // There are no add'l implied bounds when checking a
        // standalone expr (e.g., the `E` in a type like `[u32; E]`).
        rcx.outlives_environment.save_implied_bounds(id);

        if !self.errors_reported_since_creation() {
            // regionck assumes typeck succeeded
            rcx.visit_body(body);
            rcx.visit_region_obligations(id);
        }
        rcx.resolve_regions_and_report_errors(RegionckMode::for_item_body(self.tcx));
    }

    /// Region checking during the WF phase for items. `wf_tys` are the
    /// types from which we should derive implied bounds, if any.
    pub fn regionck_item(&self, item_id: hir::HirId, span: Span, wf_tys: &[Ty<'tcx>]) {
        debug!("regionck_item(item.id={:?}, wf_tys={:?})", item_id, wf_tys);
        let subject = self.tcx.hir().local_def_id(item_id);
        let mut rcx = RegionCtxt::new(self, item_id, Subject(subject), self.param_env);
        rcx.outlives_environment.add_implied_bounds(self, wf_tys, item_id, span);
        rcx.outlives_environment.save_implied_bounds(item_id);
        rcx.visit_region_obligations(item_id);
        rcx.resolve_regions_and_report_errors(RegionckMode::default());
    }

    /// Region check a function body. Not invoked on closures, but
    /// only on the "root" fn item (in which closures may be
    /// embedded). Walks the function body and adds various add'l
    /// constraints that are needed for region inference. This is
    /// separated both to isolate "pure" region constraints from the
    /// rest of type check and because sometimes we need type
    /// inference to have completed before we can determine which
    /// constraints to add.
    pub fn regionck_fn(&self, fn_id: hir::HirId, body: &'tcx hir::Body<'tcx>) {
        debug!("regionck_fn(id={})", fn_id);
        let subject = self.tcx.hir().body_owner_def_id(body.id());
        let hir_id = body.value.hir_id;
        let mut rcx = RegionCtxt::new(self, hir_id, Subject(subject), self.param_env);

        if !self.errors_reported_since_creation() {
            // regionck assumes typeck succeeded
            rcx.visit_fn_body(fn_id, body, self.tcx.hir().span(fn_id));
        }

        rcx.resolve_regions_and_report_errors(RegionckMode::for_item_body(self.tcx));
    }
}

///////////////////////////////////////////////////////////////////////////
// INTERNALS

pub struct RegionCtxt<'a, 'tcx> {
    pub fcx: &'a FnCtxt<'a, 'tcx>,

    pub region_scope_tree: &'tcx region::ScopeTree,

    outlives_environment: OutlivesEnvironment<'tcx>,

    // id of innermost fn body id
    body_id: hir::HirId,
    body_owner: LocalDefId,

    // id of AST node being analyzed (the subject of the analysis).
    subject_def_id: LocalDefId,
}

impl<'a, 'tcx> Deref for RegionCtxt<'a, 'tcx> {
    type Target = FnCtxt<'a, 'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.fcx
    }
}

pub struct Subject(LocalDefId);

impl<'a, 'tcx> RegionCtxt<'a, 'tcx> {
    pub fn new(
        fcx: &'a FnCtxt<'a, 'tcx>,
        initial_body_id: hir::HirId,
        Subject(subject): Subject,
        param_env: ty::ParamEnv<'tcx>,
    ) -> RegionCtxt<'a, 'tcx> {
        let region_scope_tree = fcx.tcx.region_scope_tree(subject);
        let outlives_environment = OutlivesEnvironment::new(param_env);
        RegionCtxt {
            fcx,
            region_scope_tree,
            body_id: initial_body_id,
            body_owner: subject,
            subject_def_id: subject,
            outlives_environment,
        }
    }

    /// Try to resolve the type for the given node, returning `t_err` if an error results. Note that
    /// we never care about the details of the error, the same error will be detected and reported
    /// in the writeback phase.
    ///
    /// Note one important point: we do not attempt to resolve *region variables* here. This is
    /// because regionck is essentially adding constraints to those region variables and so may yet
    /// influence how they are resolved.
    ///
    /// Consider this silly example:
    ///
    /// ```
    /// fn borrow(x: &i32) -> &i32 {x}
    /// fn foo(x: @i32) -> i32 {  // block: B
    ///     let b = borrow(x);    // region: <R0>
    ///     *b
    /// }
    /// ```
    ///
    /// Here, the region of `b` will be `<R0>`. `<R0>` is constrained to be some subregion of the
    /// block B and some superregion of the call. If we forced it now, we'd choose the smaller
    /// region (the call). But that would make the *b illegal. Since we don't resolve, the type
    /// of b will be `&<R0>.i32` and then `*b` will require that `<R0>` be bigger than the let and
    /// the `*b` expression, so we will effectively resolve `<R0>` to be the block B.
    pub fn resolve_type(&self, unresolved_ty: Ty<'tcx>) -> Ty<'tcx> {
        self.resolve_vars_if_possible(unresolved_ty)
    }

    /// Try to resolve the type for the given node.
    fn resolve_node_type(&self, id: hir::HirId) -> Ty<'tcx> {
        let t = self.node_ty(id);
        self.resolve_type(t)
    }

    /// This is the "main" function when region-checking a function item or a
    /// closure within a function item. It begins by updating various fields
    /// (e.g., `outlives_environment`) to be appropriate to the function and
    /// then adds constraints derived from the function body.
    ///
    /// Note that it does **not** restore the state of the fields that
    /// it updates! This is intentional, since -- for the main
    /// function -- we wish to be able to read the final
    /// `outlives_environment` and other fields from the caller. For
    /// closures, however, we save and restore any "scoped state"
    /// before we invoke this function. (See `visit_fn` in the
    /// `intravisit::Visitor` impl below.)
    fn visit_fn_body(
        &mut self,
        id: hir::HirId, // the id of the fn itself
        body: &'tcx hir::Body<'tcx>,
        span: Span,
    ) {
        // When we enter a function, we can derive
        debug!("visit_fn_body(id={:?})", id);

        let body_id = body.id();
        self.body_id = body_id.hir_id;
        self.body_owner = self.tcx.hir().body_owner_def_id(body_id);

        let fn_sig = {
            match self.typeck_results.borrow().liberated_fn_sigs().get(id) {
                Some(f) => *f,
                None => {
                    bug!("No fn-sig entry for id={:?}", id);
                }
            }
        };

        // Collect the types from which we create inferred bounds.
        // For the return type, if diverging, substitute `bool` just
        // because it will have no effect.
        //
        // FIXME(#27579) return types should not be implied bounds
        let fn_sig_tys: Vec<_> =
            fn_sig.inputs().iter().cloned().chain(Some(fn_sig.output())).collect();

        self.outlives_environment.add_implied_bounds(
            self.fcx,
            &fn_sig_tys[..],
            body_id.hir_id,
            span,
        );
        self.outlives_environment.save_implied_bounds(body_id.hir_id);
        self.link_fn_params(&body.params);
        self.visit_body(body);
        self.visit_region_obligations(body_id.hir_id);

        self.constrain_opaque_types(
            &self.fcx.opaque_types.borrow(),
            self.outlives_environment.free_region_map(),
        );
    }

    fn visit_region_obligations(&mut self, hir_id: hir::HirId) {
        debug!("visit_region_obligations: hir_id={:?}", hir_id);

        // region checking can introduce new pending obligations
        // which, when processed, might generate new region
        // obligations. So make sure we process those.
        self.select_all_obligations_or_error();
    }

    fn resolve_regions_and_report_errors(&self, mode: RegionckMode) {
        self.infcx.process_registered_region_obligations(
            self.outlives_environment.region_bound_pairs_map(),
            Some(self.tcx.lifetimes.re_root_empty),
            self.param_env,
        );

        self.fcx.resolve_regions_and_report_errors(
            self.subject_def_id.to_def_id(),
            &self.outlives_environment,
            mode,
        );
    }

    fn constrain_bindings_in_pat(&mut self, pat: &hir::Pat<'_>) {
        debug!("regionck::visit_pat(pat={:?})", pat);
        pat.each_binding(|_, hir_id, span, _| {
            let typ = self.resolve_node_type(hir_id);
            let body_id = self.body_id;
            let _ = dropck::check_drop_obligations(self, typ, span, body_id);
        })
    }
}

impl<'a, 'tcx> Visitor<'tcx> for RegionCtxt<'a, 'tcx> {
    // (..) FIXME(#3238) should use visit_pat, not visit_arm/visit_local,
    // However, right now we run into an issue whereby some free
    // regions are not properly related if they appear within the
    // types of arguments that must be inferred. This could be
    // addressed by deferring the construction of the region
    // hierarchy, and in particular the relationships between free
    // regions, until regionck, as described in #3238.

    type Map = intravisit::ErasedMap<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }

    fn visit_fn(
        &mut self,
        fk: intravisit::FnKind<'tcx>,
        _: &'tcx hir::FnDecl<'tcx>,
        body_id: hir::BodyId,
        span: Span,
        hir_id: hir::HirId,
    ) {
        assert!(
            matches!(fk, intravisit::FnKind::Closure(..)),
            "visit_fn invoked for something other than a closure"
        );

        // Save state of current function before invoking
        // `visit_fn_body`.  We will restore afterwards.
        let old_body_id = self.body_id;
        let old_body_owner = self.body_owner;
        let env_snapshot = self.outlives_environment.push_snapshot_pre_closure();

        let body = self.tcx.hir().body(body_id);
        self.visit_fn_body(hir_id, body, span);

        // Restore state from previous function.
        self.outlives_environment.pop_snapshot_post_closure(env_snapshot);
        self.body_id = old_body_id;
        self.body_owner = old_body_owner;
    }

    //visit_pat: visit_pat, // (..) see above

    fn visit_arm(&mut self, arm: &'tcx hir::Arm<'tcx>) {
        // see above
        self.constrain_bindings_in_pat(&arm.pat);
        intravisit::walk_arm(self, arm);
    }

    fn visit_local(&mut self, l: &'tcx hir::Local<'tcx>) {
        // see above
        self.constrain_bindings_in_pat(&l.pat);
        self.link_local(l);
        intravisit::walk_local(self, l);
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        // Check any autoderefs or autorefs that appear.
        let cmt_result = self.constrain_adjustments(expr);

        // If necessary, constrain destructors in this expression. This will be
        // the adjusted form if there is an adjustment.
        match cmt_result {
            Ok(head_cmt) => {
                self.check_safety_of_rvalue_destructor_if_necessary(&head_cmt, expr.span);
            }
            Err(..) => {
                self.tcx.sess.delay_span_bug(expr.span, "cat_expr Errd");
            }
        }

        match expr.kind {
            hir::ExprKind::AddrOf(hir::BorrowKind::Ref, m, ref base) => {
                self.link_addr_of(expr, m, &base);

                intravisit::walk_expr(self, expr);
            }

            hir::ExprKind::Match(ref discr, ref arms, _) => {
                self.link_match(&discr, &arms[..]);

                intravisit::walk_expr(self, expr);
            }

            _ => intravisit::walk_expr(self, expr),
        }
    }
}

impl<'a, 'tcx> RegionCtxt<'a, 'tcx> {
    /// Creates a temporary `MemCategorizationContext` and pass it to the closure.
    fn with_mc<F, R>(&self, f: F) -> R
    where
        F: for<'b> FnOnce(mc::MemCategorizationContext<'b, 'tcx>) -> R,
    {
        f(mc::MemCategorizationContext::new(
            &self.infcx,
            self.outlives_environment.param_env,
            self.body_owner,
            &self.typeck_results.borrow(),
        ))
    }

    /// Invoked on any adjustments that occur. Checks that if this is a region pointer being
    /// dereferenced, the lifetime of the pointer includes the deref expr.
    fn constrain_adjustments(
        &mut self,
        expr: &hir::Expr<'_>,
    ) -> mc::McResult<PlaceWithHirId<'tcx>> {
        debug!("constrain_adjustments(expr={:?})", expr);

        let mut place = self.with_mc(|mc| mc.cat_expr_unadjusted(expr))?;

        let typeck_results = self.typeck_results.borrow();
        let adjustments = typeck_results.expr_adjustments(&expr);
        if adjustments.is_empty() {
            return Ok(place);
        }

        debug!("constrain_adjustments: adjustments={:?}", adjustments);

        // If necessary, constrain destructors in the unadjusted form of this
        // expression.
        self.check_safety_of_rvalue_destructor_if_necessary(&place, expr.span);

        for adjustment in adjustments {
            debug!("constrain_adjustments: adjustment={:?}, place={:?}", adjustment, place);

            if let adjustment::Adjust::Deref(Some(deref)) = adjustment.kind {
                self.link_region(
                    expr.span,
                    deref.region,
                    ty::BorrowKind::from_mutbl(deref.mutbl),
                    &place,
                );
            }

            if let adjustment::Adjust::Borrow(ref autoref) = adjustment.kind {
                self.link_autoref(expr, &place, autoref);
            }

            place = self.with_mc(|mc| mc.cat_expr_adjusted(expr, place, &adjustment))?;
        }

        Ok(place)
    }

    fn check_safety_of_rvalue_destructor_if_necessary(
        &mut self,
        place_with_id: &PlaceWithHirId<'tcx>,
        span: Span,
    ) {
        if let PlaceBase::Rvalue = place_with_id.place.base {
            if place_with_id.place.projections.is_empty() {
                let typ = self.resolve_type(place_with_id.place.ty());
                let body_id = self.body_id;
                let _ = dropck::check_drop_obligations(self, typ, span, body_id);
            }
        }
    }
    /// Adds constraints to inference such that `T: 'a` holds (or
    /// reports an error if it cannot).
    ///
    /// # Parameters
    ///
    /// - `origin`, the reason we need this constraint
    /// - `ty`, the type `T`
    /// - `region`, the region `'a`
    pub fn type_must_outlive(
        &self,
        origin: infer::SubregionOrigin<'tcx>,
        ty: Ty<'tcx>,
        region: ty::Region<'tcx>,
    ) {
        self.infcx.register_region_obligation(
            self.body_id,
            RegionObligation { sub_region: region, sup_type: ty, origin },
        );
    }

    /// Computes the guarantor for an expression `&base` and then ensures that the lifetime of the
    /// resulting pointer is linked to the lifetime of its guarantor (if any).
    fn link_addr_of(
        &mut self,
        expr: &hir::Expr<'_>,
        mutability: hir::Mutability,
        base: &hir::Expr<'_>,
    ) {
        debug!("link_addr_of(expr={:?}, base={:?})", expr, base);

        let cmt = ignore_err!(self.with_mc(|mc| mc.cat_expr(base)));

        debug!("link_addr_of: cmt={:?}", cmt);

        self.link_region_from_node_type(expr.span, expr.hir_id, mutability, &cmt);
    }

    /// Computes the guarantors for any ref bindings in a `let` and
    /// then ensures that the lifetime of the resulting pointer is
    /// linked to the lifetime of the initialization expression.
    fn link_local(&self, local: &hir::Local<'_>) {
        debug!("regionck::for_local()");
        let init_expr = match local.init {
            None => {
                return;
            }
            Some(ref expr) => &**expr,
        };
        let discr_cmt = ignore_err!(self.with_mc(|mc| mc.cat_expr(init_expr)));
        self.link_pattern(discr_cmt, &local.pat);
    }

    /// Computes the guarantors for any ref bindings in a match and
    /// then ensures that the lifetime of the resulting pointer is
    /// linked to the lifetime of its guarantor (if any).
    fn link_match(&self, discr: &hir::Expr<'_>, arms: &[hir::Arm<'_>]) {
        debug!("regionck::for_match()");
        let discr_cmt = ignore_err!(self.with_mc(|mc| mc.cat_expr(discr)));
        debug!("discr_cmt={:?}", discr_cmt);
        for arm in arms {
            self.link_pattern(discr_cmt.clone(), &arm.pat);
        }
    }

    /// Computes the guarantors for any ref bindings in a match and
    /// then ensures that the lifetime of the resulting pointer is
    /// linked to the lifetime of its guarantor (if any).
    fn link_fn_params(&self, params: &[hir::Param<'_>]) {
        for param in params {
            let param_ty = self.node_ty(param.hir_id);
            let param_cmt =
                self.with_mc(|mc| mc.cat_rvalue(param.hir_id, param.pat.span, param_ty));
            debug!("param_ty={:?} param_cmt={:?} param={:?}", param_ty, param_cmt, param);
            self.link_pattern(param_cmt, &param.pat);
        }
    }

    /// Link lifetimes of any ref bindings in `root_pat` to the pointers found
    /// in the discriminant, if needed.
    fn link_pattern(&self, discr_cmt: PlaceWithHirId<'tcx>, root_pat: &hir::Pat<'_>) {
        debug!("link_pattern(discr_cmt={:?}, root_pat={:?})", discr_cmt, root_pat);
        ignore_err!(self.with_mc(|mc| {
            mc.cat_pattern(discr_cmt, root_pat, |sub_cmt, hir::Pat { kind, span, hir_id, .. }| {
                // `ref x` pattern
                if let PatKind::Binding(..) = kind {
                    if let Some(ty::BindByReference(mutbl)) =
                        mc.typeck_results.extract_binding_mode(self.tcx.sess, *hir_id, *span)
                    {
                        self.link_region_from_node_type(*span, *hir_id, mutbl, &sub_cmt);
                    }
                }
            })
        }));
    }

    /// Link lifetime of borrowed pointer resulting from autoref to lifetimes in the value being
    /// autoref'd.
    fn link_autoref(
        &self,
        expr: &hir::Expr<'_>,
        expr_cmt: &PlaceWithHirId<'tcx>,
        autoref: &adjustment::AutoBorrow<'tcx>,
    ) {
        debug!("link_autoref(autoref={:?}, expr_cmt={:?})", autoref, expr_cmt);

        match *autoref {
            adjustment::AutoBorrow::Ref(r, m) => {
                self.link_region(expr.span, r, ty::BorrowKind::from_mutbl(m.into()), expr_cmt);
            }

            adjustment::AutoBorrow::RawPtr(_) => {}
        }
    }

    /// Like `link_region()`, except that the region is extracted from the type of `id`,
    /// which must be some reference (`&T`, `&str`, etc).
    fn link_region_from_node_type(
        &self,
        span: Span,
        id: hir::HirId,
        mutbl: hir::Mutability,
        cmt_borrowed: &PlaceWithHirId<'tcx>,
    ) {
        debug!(
            "link_region_from_node_type(id={:?}, mutbl={:?}, cmt_borrowed={:?})",
            id, mutbl, cmt_borrowed
        );

        let rptr_ty = self.resolve_node_type(id);
        if let ty::Ref(r, _, _) = rptr_ty.kind() {
            debug!("rptr_ty={}", rptr_ty);
            self.link_region(span, r, ty::BorrowKind::from_mutbl(mutbl), cmt_borrowed);
        }
    }

    /// Informs the inference engine that `borrow_cmt` is being borrowed with
    /// kind `borrow_kind` and lifetime `borrow_region`.
    /// In order to ensure borrowck is satisfied, this may create constraints
    /// between regions, as explained in `link_reborrowed_region()`.
    fn link_region(
        &self,
        span: Span,
        borrow_region: ty::Region<'tcx>,
        borrow_kind: ty::BorrowKind,
        borrow_place: &PlaceWithHirId<'tcx>,
    ) {
        let origin = infer::DataBorrowed(borrow_place.place.ty(), span);
        self.type_must_outlive(origin, borrow_place.place.ty(), borrow_region);

        for pointer_ty in borrow_place.place.deref_tys() {
            debug!(
                "link_region(borrow_region={:?}, borrow_kind={:?}, pointer_ty={:?})",
                borrow_region, borrow_kind, borrow_place
            );
            match *pointer_ty.kind() {
                ty::RawPtr(_) => return,
                ty::Ref(ref_region, _, ref_mutability) => {
                    if self.link_reborrowed_region(span, borrow_region, ref_region, ref_mutability)
                    {
                        return;
                    }
                }
                _ => assert!(pointer_ty.is_box(), "unexpected built-in deref type {}", pointer_ty),
            }
        }
        if let PlaceBase::Upvar(upvar_id) = borrow_place.place.base {
            self.link_upvar_region(span, borrow_region, upvar_id);
        }
    }

    /// This is the most complicated case: the path being borrowed is
    /// itself the referent of a borrowed pointer. Let me give an
    /// example fragment of code to make clear(er) the situation:
    ///
    /// ```ignore (incomplete Rust code)
    /// let r: &'a mut T = ...;  // the original reference "r" has lifetime 'a
    /// ...
    /// &'z *r                   // the reborrow has lifetime 'z
    /// ```
    ///
    /// Now, in this case, our primary job is to add the inference
    /// constraint that `'z <= 'a`. Given this setup, let's clarify the
    /// parameters in (roughly) terms of the example:
    ///
    /// ```plain,ignore (pseudo-Rust)
    /// A borrow of: `& 'z bk * r` where `r` has type `& 'a bk T`
    /// borrow_region   ^~                 ref_region    ^~
    /// borrow_kind        ^~               ref_kind        ^~
    /// ref_cmt                 ^
    /// ```
    ///
    /// Here `bk` stands for some borrow-kind (e.g., `mut`, `uniq`, etc).
    ///
    /// There is a complication beyond the simple scenario I just painted: there
    /// may in fact be more levels of reborrowing. In the example, I said the
    /// borrow was like `&'z *r`, but it might in fact be a borrow like
    /// `&'z **q` where `q` has type `&'a &'b mut T`. In that case, we want to
    /// ensure that `'z <= 'a` and `'z <= 'b`.
    ///
    /// The return value of this function indicates whether we *don't* need to
    /// the recurse to the next reference up.
    ///
    /// This is explained more below.
    fn link_reborrowed_region(
        &self,
        span: Span,
        borrow_region: ty::Region<'tcx>,
        ref_region: ty::Region<'tcx>,
        ref_mutability: hir::Mutability,
    ) -> bool {
        debug!("link_reborrowed_region: {:?} <= {:?}", borrow_region, ref_region);
        self.sub_regions(infer::Reborrow(span), borrow_region, ref_region);

        // Decide whether we need to recurse and link any regions within
        // the `ref_cmt`. This is concerned for the case where the value
        // being reborrowed is in fact a borrowed pointer found within
        // another borrowed pointer. For example:
        //
        //    let p: &'b &'a mut T = ...;
        //    ...
        //    &'z **p
        //
        // What makes this case particularly tricky is that, if the data
        // being borrowed is a `&mut` or `&uniq` borrow, borrowck requires
        // not only that `'z <= 'a`, (as before) but also `'z <= 'b`
        // (otherwise the user might mutate through the `&mut T` reference
        // after `'b` expires and invalidate the borrow we are looking at
        // now).
        //
        // So let's re-examine our parameters in light of this more
        // complicated (possible) scenario:
        //
        //     A borrow of: `& 'z bk * * p` where `p` has type `&'b bk & 'a bk T`
        //     borrow_region   ^~                 ref_region             ^~
        //     borrow_kind        ^~               ref_kind                 ^~
        //     ref_cmt                 ^~~
        //
        // (Note that since we have not examined `ref_cmt.cat`, we don't
        // know whether this scenario has occurred; but I wanted to show
        // how all the types get adjusted.)
        match ref_mutability {
            hir::Mutability::Not => {
                // The reference being reborrowed is a shareable ref of
                // type `&'a T`. In this case, it doesn't matter where we
                // *found* the `&T` pointer, the memory it references will
                // be valid and immutable for `'a`. So we can stop here.
                true
            }

            hir::Mutability::Mut => {
                // The reference being reborrowed is either an `&mut T`. This is
                // the case where recursion is needed.
                false
            }
        }
    }

    /// An upvar may be behind up to 2 references:
    ///
    /// * One can come from the reference to a "by-reference" upvar.
    /// * Another one can come from the reference to the closure itself if it's
    ///   a `FnMut` or `Fn` closure.
    ///
    /// This function links the lifetimes of those references to the lifetime
    /// of the borrow that's provided. See [RegionCtxt::link_reborrowed_region] for some
    /// more explanation of this in the general case.
    ///
    /// We also supply a *cause*, and in this case we set the cause to
    /// indicate that the reference being "reborrowed" is itself an upvar. This
    /// provides a nicer error message should something go wrong.
    fn link_upvar_region(
        &self,
        span: Span,
        borrow_region: ty::Region<'tcx>,
        upvar_id: ty::UpvarId,
    ) {
        debug!("link_upvar_region(borrorw_region={:?}, upvar_id={:?}", borrow_region, upvar_id);
        // A by-reference upvar can't be borrowed for longer than the
        // upvar is borrowed from the environment.
        match self.typeck_results.borrow().upvar_capture(upvar_id) {
            ty::UpvarCapture::ByRef(upvar_borrow) => {
                self.sub_regions(
                    infer::ReborrowUpvar(span, upvar_id),
                    borrow_region,
                    upvar_borrow.region,
                );
                if let ty::ImmBorrow = upvar_borrow.kind {
                    debug!("link_upvar_region: capture by shared ref");
                    return;
                }
            }
            ty::UpvarCapture::ByValue(_) => {}
        }
        let fn_hir_id = self.tcx.hir().local_def_id_to_hir_id(upvar_id.closure_expr_id);
        let ty = self.resolve_node_type(fn_hir_id);
        debug!("link_upvar_region: ty={:?}", ty);

        // A closure capture can't be borrowed for longer than the
        // reference to the closure.
        if let ty::Closure(_, substs) = ty.kind() {
            match self.infcx.closure_kind(substs) {
                Some(ty::ClosureKind::Fn | ty::ClosureKind::FnMut) => {
                    // Region of environment pointer
                    let env_region = self.tcx.mk_region(ty::ReFree(ty::FreeRegion {
                        scope: upvar_id.closure_expr_id.to_def_id(),
                        bound_region: ty::BrEnv,
                    }));
                    self.sub_regions(
                        infer::ReborrowUpvar(span, upvar_id),
                        borrow_region,
                        env_region,
                    );
                }
                Some(ty::ClosureKind::FnOnce) => {}
                None => {
                    span_bug!(span, "Have not inferred closure kind before regionck");
                }
            }
        }
    }
}
