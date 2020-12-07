//! A different sort of visitor for walking fn bodies. Unlike the
//! normal visitor, which just walks the entire body in one shot, the
//! `ExprUseVisitor` determines how expressions are being used.

pub use self::ConsumeMode::*;

// Export these here so that Clippy can use them.
pub use rustc_middle::hir::place::{PlaceBase, PlaceWithHirId, Projection};

use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::PatKind;
use rustc_index::vec::Idx;
use rustc_infer::infer::InferCtxt;
use rustc_middle::hir::place::ProjectionKind;
use rustc_middle::ty::{self, adjustment, TyCtxt};
use rustc_span::Span;
use rustc_target::abi::VariantIdx;

use crate::mem_categorization as mc;

///////////////////////////////////////////////////////////////////////////
// The Delegate trait

/// This trait defines the callbacks you can expect to receive when
/// employing the ExprUseVisitor.
pub trait Delegate<'tcx> {
    // The value found at `place` is either copied or moved, depending
    // on `mode`. Where `diag_expr_id` is the id used for diagnostics for `place`.
    //
    // The parameter `diag_expr_id` indicates the HIR id that ought to be used for
    // diagnostics. Around pattern matching such as `let pat = expr`, the diagnostic
    // id will be the id of the expression `expr` but the place itself will have
    // the id of the binding in the pattern `pat`.
    fn consume(
        &mut self,
        place_with_id: &PlaceWithHirId<'tcx>,
        diag_expr_id: hir::HirId,
        mode: ConsumeMode,
    );

    // The value found at `place` is being borrowed with kind `bk`.
    // `diag_expr_id` is the id used for diagnostics (see `consume` for more details).
    fn borrow(
        &mut self,
        place_with_id: &PlaceWithHirId<'tcx>,
        diag_expr_id: hir::HirId,
        bk: ty::BorrowKind,
    );

    // The path at `assignee_place` is being assigned to.
    // `diag_expr_id` is the id used for diagnostics (see `consume` for more details).
    fn mutate(&mut self, assignee_place: &PlaceWithHirId<'tcx>, diag_expr_id: hir::HirId);
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum ConsumeMode {
    Copy, // reference to x where x has a type that copies
    Move, // reference to x where x has a type that moves
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum MutateMode {
    Init,
    JustWrite,    // x = y
    WriteAndRead, // x += y
}

///////////////////////////////////////////////////////////////////////////
// The ExprUseVisitor type
//
// This is the code that actually walks the tree.
pub struct ExprUseVisitor<'a, 'tcx> {
    mc: mc::MemCategorizationContext<'a, 'tcx>,
    body_owner: LocalDefId,
    delegate: &'a mut dyn Delegate<'tcx>,
}

// If the MC results in an error, it's because the type check
// failed (or will fail, when the error is uncovered and reported
// during writeback). In this case, we just ignore this part of the
// code.
//
// Note that this macro appears similar to try!(), but, unlike try!(),
// it does not propagate the error.
macro_rules! return_if_err {
    ($inp: expr) => {
        match $inp {
            Ok(v) => v,
            Err(()) => {
                debug!("mc reported err");
                return;
            }
        }
    };
}

impl<'a, 'tcx> ExprUseVisitor<'a, 'tcx> {
    /// Creates the ExprUseVisitor, configuring it with the various options provided:
    ///
    /// - `delegate` -- who receives the callbacks
    /// - `param_env` --- parameter environment for trait lookups (esp. pertaining to `Copy`)
    /// - `typeck_results` --- typeck results for the code being analyzed
    pub fn new(
        delegate: &'a mut (dyn Delegate<'tcx> + 'a),
        infcx: &'a InferCtxt<'a, 'tcx>,
        body_owner: LocalDefId,
        param_env: ty::ParamEnv<'tcx>,
        typeck_results: &'a ty::TypeckResults<'tcx>,
    ) -> Self {
        ExprUseVisitor {
            mc: mc::MemCategorizationContext::new(infcx, param_env, body_owner, typeck_results),
            body_owner,
            delegate,
        }
    }

    pub fn consume_body(&mut self, body: &hir::Body<'_>) {
        debug!("consume_body(body={:?})", body);

        for param in body.params {
            let param_ty = return_if_err!(self.mc.pat_ty_adjusted(&param.pat));
            debug!("consume_body: param_ty = {:?}", param_ty);

            let param_place = self.mc.cat_rvalue(param.hir_id, param.pat.span, param_ty);

            self.walk_irrefutable_pat(&param_place, &param.pat);
        }

        self.consume_expr(&body.value);
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.mc.tcx()
    }

    fn delegate_consume(&mut self, place_with_id: &PlaceWithHirId<'tcx>, diag_expr_id: hir::HirId) {
        debug!("delegate_consume(place_with_id={:?})", place_with_id);

        let mode = copy_or_move(&self.mc, place_with_id);
        self.delegate.consume(place_with_id, diag_expr_id, mode);
    }

    fn consume_exprs(&mut self, exprs: &[hir::Expr<'_>]) {
        for expr in exprs {
            self.consume_expr(&expr);
        }
    }

    pub fn consume_expr(&mut self, expr: &hir::Expr<'_>) {
        debug!("consume_expr(expr={:?})", expr);

        let place_with_id = return_if_err!(self.mc.cat_expr(expr));
        self.delegate_consume(&place_with_id, place_with_id.hir_id);
        self.walk_expr(expr);
    }

    fn mutate_expr(&mut self, expr: &hir::Expr<'_>) {
        let place_with_id = return_if_err!(self.mc.cat_expr(expr));
        self.delegate.mutate(&place_with_id, place_with_id.hir_id);
        self.walk_expr(expr);
    }

    fn borrow_expr(&mut self, expr: &hir::Expr<'_>, bk: ty::BorrowKind) {
        debug!("borrow_expr(expr={:?}, bk={:?})", expr, bk);

        let place_with_id = return_if_err!(self.mc.cat_expr(expr));
        self.delegate.borrow(&place_with_id, place_with_id.hir_id, bk);

        self.walk_expr(expr)
    }

    fn select_from_expr(&mut self, expr: &hir::Expr<'_>) {
        self.walk_expr(expr)
    }

    pub fn walk_expr(&mut self, expr: &hir::Expr<'_>) {
        debug!("walk_expr(expr={:?})", expr);

        self.walk_adjustment(expr);

        match expr.kind {
            hir::ExprKind::Path(_) => {}

            hir::ExprKind::Type(ref subexpr, _) => self.walk_expr(subexpr),

            hir::ExprKind::Unary(hir::UnOp::UnDeref, ref base) => {
                // *base
                self.select_from_expr(base);
            }

            hir::ExprKind::Field(ref base, _) => {
                // base.f
                self.select_from_expr(base);
            }

            hir::ExprKind::Index(ref lhs, ref rhs) => {
                // lhs[rhs]
                self.select_from_expr(lhs);
                self.consume_expr(rhs);
            }

            hir::ExprKind::Call(ref callee, ref args) => {
                // callee(args)
                self.consume_expr(callee);
                self.consume_exprs(args);
            }

            hir::ExprKind::MethodCall(.., ref args, _) => {
                // callee.m(args)
                self.consume_exprs(args);
            }

            hir::ExprKind::Struct(_, ref fields, ref opt_with) => {
                self.walk_struct_expr(fields, opt_with);
            }

            hir::ExprKind::Tup(ref exprs) => {
                self.consume_exprs(exprs);
            }

            hir::ExprKind::Match(ref discr, arms, _) => {
                let discr_place = return_if_err!(self.mc.cat_expr(&discr));
                self.borrow_expr(&discr, ty::ImmBorrow);

                // treatment of the discriminant is handled while walking the arms.
                for arm in arms {
                    self.walk_arm(&discr_place, arm);
                }
            }

            hir::ExprKind::Array(ref exprs) => {
                self.consume_exprs(exprs);
            }

            hir::ExprKind::AddrOf(_, m, ref base) => {
                // &base
                // make sure that the thing we are pointing out stays valid
                // for the lifetime `scope_r` of the resulting ptr:
                let bk = ty::BorrowKind::from_mutbl(m);
                self.borrow_expr(&base, bk);
            }

            hir::ExprKind::InlineAsm(ref asm) => {
                for (op, _op_sp) in asm.operands {
                    match op {
                        hir::InlineAsmOperand::In { expr, .. }
                        | hir::InlineAsmOperand::Const { expr, .. }
                        | hir::InlineAsmOperand::Sym { expr, .. } => self.consume_expr(expr),
                        hir::InlineAsmOperand::Out { expr, .. } => {
                            if let Some(expr) = expr {
                                self.mutate_expr(expr);
                            }
                        }
                        hir::InlineAsmOperand::InOut { expr, .. } => {
                            self.mutate_expr(expr);
                        }
                        hir::InlineAsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                            self.consume_expr(in_expr);
                            if let Some(out_expr) = out_expr {
                                self.mutate_expr(out_expr);
                            }
                        }
                    }
                }
            }

            hir::ExprKind::LlvmInlineAsm(ref ia) => {
                for (o, output) in ia.inner.outputs.iter().zip(ia.outputs_exprs) {
                    if o.is_indirect {
                        self.consume_expr(output);
                    } else {
                        self.mutate_expr(output);
                    }
                }
                self.consume_exprs(&ia.inputs_exprs);
            }

            hir::ExprKind::Continue(..)
            | hir::ExprKind::Lit(..)
            | hir::ExprKind::ConstBlock(..)
            | hir::ExprKind::Err => {}

            hir::ExprKind::Loop(ref blk, _, _) => {
                self.walk_block(blk);
            }

            hir::ExprKind::Unary(_, ref lhs) => {
                self.consume_expr(lhs);
            }

            hir::ExprKind::Binary(_, ref lhs, ref rhs) => {
                self.consume_expr(lhs);
                self.consume_expr(rhs);
            }

            hir::ExprKind::Block(ref blk, _) => {
                self.walk_block(blk);
            }

            hir::ExprKind::Break(_, ref opt_expr) | hir::ExprKind::Ret(ref opt_expr) => {
                if let Some(ref expr) = *opt_expr {
                    self.consume_expr(expr);
                }
            }

            hir::ExprKind::Assign(ref lhs, ref rhs, _) => {
                self.mutate_expr(lhs);
                self.consume_expr(rhs);
            }

            hir::ExprKind::Cast(ref base, _) => {
                self.consume_expr(base);
            }

            hir::ExprKind::DropTemps(ref expr) => {
                self.consume_expr(expr);
            }

            hir::ExprKind::AssignOp(_, ref lhs, ref rhs) => {
                if self.mc.typeck_results.is_method_call(expr) {
                    self.consume_expr(lhs);
                } else {
                    self.mutate_expr(lhs);
                }
                self.consume_expr(rhs);
            }

            hir::ExprKind::Repeat(ref base, _) => {
                self.consume_expr(base);
            }

            hir::ExprKind::Closure(..) => {
                self.walk_captures(expr);
            }

            hir::ExprKind::Box(ref base) => {
                self.consume_expr(base);
            }

            hir::ExprKind::Yield(ref value, _) => {
                self.consume_expr(value);
            }
        }
    }

    fn walk_stmt(&mut self, stmt: &hir::Stmt<'_>) {
        match stmt.kind {
            hir::StmtKind::Local(ref local) => {
                self.walk_local(&local);
            }

            hir::StmtKind::Item(_) => {
                // We don't visit nested items in this visitor,
                // only the fn body we were given.
            }

            hir::StmtKind::Expr(ref expr) | hir::StmtKind::Semi(ref expr) => {
                self.consume_expr(&expr);
            }
        }
    }

    fn walk_local(&mut self, local: &hir::Local<'_>) {
        if let Some(ref expr) = local.init {
            // Variable declarations with
            // initializers are considered
            // "assigns", which is handled by
            // `walk_pat`:
            self.walk_expr(&expr);
            let init_place = return_if_err!(self.mc.cat_expr(&expr));
            self.walk_irrefutable_pat(&init_place, &local.pat);
        }
    }

    /// Indicates that the value of `blk` will be consumed, meaning either copied or moved
    /// depending on its type.
    fn walk_block(&mut self, blk: &hir::Block<'_>) {
        debug!("walk_block(blk.hir_id={})", blk.hir_id);

        for stmt in blk.stmts {
            self.walk_stmt(stmt);
        }

        if let Some(ref tail_expr) = blk.expr {
            self.consume_expr(&tail_expr);
        }
    }

    fn walk_struct_expr(
        &mut self,
        fields: &[hir::Field<'_>],
        opt_with: &Option<&'hir hir::Expr<'_>>,
    ) {
        // Consume the expressions supplying values for each field.
        for field in fields {
            self.consume_expr(&field.expr);
        }

        let with_expr = match *opt_with {
            Some(ref w) => &**w,
            None => {
                return;
            }
        };

        let with_place = return_if_err!(self.mc.cat_expr(&with_expr));

        // Select just those fields of the `with`
        // expression that will actually be used
        match with_place.place.ty().kind() {
            ty::Adt(adt, substs) if adt.is_struct() => {
                // Consume those fields of the with expression that are needed.
                for (f_index, with_field) in adt.non_enum_variant().fields.iter().enumerate() {
                    let is_mentioned = fields.iter().any(|f| {
                        self.tcx().field_index(f.hir_id, self.mc.typeck_results) == f_index
                    });
                    if !is_mentioned {
                        let field_place = self.mc.cat_projection(
                            &*with_expr,
                            with_place.clone(),
                            with_field.ty(self.tcx(), substs),
                            ProjectionKind::Field(f_index as u32, VariantIdx::new(0)),
                        );
                        self.delegate_consume(&field_place, field_place.hir_id);
                    }
                }
            }
            _ => {
                // the base expression should always evaluate to a
                // struct; however, when EUV is run during typeck, it
                // may not. This will generate an error earlier in typeck,
                // so we can just ignore it.
                if !self.tcx().sess.has_errors() {
                    span_bug!(with_expr.span, "with expression doesn't evaluate to a struct");
                }
            }
        }

        // walk the with expression so that complex expressions
        // are properly handled.
        self.walk_expr(with_expr);
    }

    // Invoke the appropriate delegate calls for anything that gets
    // consumed or borrowed as part of the automatic adjustment
    // process.
    fn walk_adjustment(&mut self, expr: &hir::Expr<'_>) {
        let adjustments = self.mc.typeck_results.expr_adjustments(expr);
        let mut place_with_id = return_if_err!(self.mc.cat_expr_unadjusted(expr));
        for adjustment in adjustments {
            debug!("walk_adjustment expr={:?} adj={:?}", expr, adjustment);
            match adjustment.kind {
                adjustment::Adjust::NeverToAny | adjustment::Adjust::Pointer(_) => {
                    // Creating a closure/fn-pointer or unsizing consumes
                    // the input and stores it into the resulting rvalue.
                    self.delegate_consume(&place_with_id, place_with_id.hir_id);
                }

                adjustment::Adjust::Deref(None) => {}

                // Autoderefs for overloaded Deref calls in fact reference
                // their receiver. That is, if we have `(*x)` where `x`
                // is of type `Rc<T>`, then this in fact is equivalent to
                // `x.deref()`. Since `deref()` is declared with `&self`,
                // this is an autoref of `x`.
                adjustment::Adjust::Deref(Some(ref deref)) => {
                    let bk = ty::BorrowKind::from_mutbl(deref.mutbl);
                    self.delegate.borrow(&place_with_id, place_with_id.hir_id, bk);
                }

                adjustment::Adjust::Borrow(ref autoref) => {
                    self.walk_autoref(expr, &place_with_id, autoref);
                }
            }
            place_with_id =
                return_if_err!(self.mc.cat_expr_adjusted(expr, place_with_id, &adjustment));
        }
    }

    /// Walks the autoref `autoref` applied to the autoderef'd
    /// `expr`. `base_place` is the mem-categorized form of `expr`
    /// after all relevant autoderefs have occurred.
    fn walk_autoref(
        &mut self,
        expr: &hir::Expr<'_>,
        base_place: &PlaceWithHirId<'tcx>,
        autoref: &adjustment::AutoBorrow<'tcx>,
    ) {
        debug!(
            "walk_autoref(expr.hir_id={} base_place={:?} autoref={:?})",
            expr.hir_id, base_place, autoref
        );

        match *autoref {
            adjustment::AutoBorrow::Ref(_, m) => {
                self.delegate.borrow(
                    base_place,
                    base_place.hir_id,
                    ty::BorrowKind::from_mutbl(m.into()),
                );
            }

            adjustment::AutoBorrow::RawPtr(m) => {
                debug!("walk_autoref: expr.hir_id={} base_place={:?}", expr.hir_id, base_place);

                self.delegate.borrow(base_place, base_place.hir_id, ty::BorrowKind::from_mutbl(m));
            }
        }
    }

    fn walk_arm(&mut self, discr_place: &PlaceWithHirId<'tcx>, arm: &hir::Arm<'_>) {
        self.walk_pat(discr_place, &arm.pat);

        if let Some(hir::Guard::If(ref e)) = arm.guard {
            self.consume_expr(e)
        }

        self.consume_expr(&arm.body);
    }

    /// Walks a pat that occurs in isolation (i.e., top-level of fn argument or
    /// let binding, and *not* a match arm or nested pat.)
    fn walk_irrefutable_pat(&mut self, discr_place: &PlaceWithHirId<'tcx>, pat: &hir::Pat<'_>) {
        self.walk_pat(discr_place, pat);
    }

    /// The core driver for walking a pattern
    fn walk_pat(&mut self, discr_place: &PlaceWithHirId<'tcx>, pat: &hir::Pat<'_>) {
        debug!("walk_pat(discr_place={:?}, pat={:?})", discr_place, pat);

        let tcx = self.tcx();
        let ExprUseVisitor { ref mc, body_owner: _, ref mut delegate } = *self;
        return_if_err!(mc.cat_pattern(discr_place.clone(), pat, |place, pat| {
            if let PatKind::Binding(_, canonical_id, ..) = pat.kind {
                debug!("walk_pat: binding place={:?} pat={:?}", place, pat,);
                if let Some(bm) =
                    mc.typeck_results.extract_binding_mode(tcx.sess, pat.hir_id, pat.span)
                {
                    debug!("walk_pat: pat.hir_id={:?} bm={:?}", pat.hir_id, bm);

                    // pat_ty: the type of the binding being produced.
                    let pat_ty = return_if_err!(mc.node_ty(pat.hir_id));
                    debug!("walk_pat: pat_ty={:?}", pat_ty);

                    // Each match binding is effectively an assignment to the
                    // binding being produced.
                    let def = Res::Local(canonical_id);
                    if let Ok(ref binding_place) = mc.cat_res(pat.hir_id, pat.span, pat_ty, def) {
                        delegate.mutate(binding_place, binding_place.hir_id);
                    }

                    // It is also a borrow or copy/move of the value being matched.
                    // In a cases of pattern like `let pat = upvar`, don't use the span
                    // of the pattern, as this just looks confusing, instead use the span
                    // of the discriminant.
                    match bm {
                        ty::BindByReference(m) => {
                            let bk = ty::BorrowKind::from_mutbl(m);
                            delegate.borrow(place, discr_place.hir_id, bk);
                        }
                        ty::BindByValue(..) => {
                            let mode = copy_or_move(mc, &place);
                            debug!("walk_pat binding consuming pat");
                            delegate.consume(place, discr_place.hir_id, mode);
                        }
                    }
                }
            }
        }));
    }

    /// Walk closure captures but using `closure_caputes` instead
    /// of `closure_min_captures`.
    ///
    /// This is needed because clippy uses `ExprUseVisitor` after TypeckResults
    /// are written back. We don't currently writeback min_captures to
    /// TypeckResults.
    fn walk_captures_closure_captures(&mut self, closure_expr: &hir::Expr<'_>) {
        // FIXME(arora-aman): Remove this function once rust-lang/project-rfc-2229#18
        // is completed.
        debug!("walk_captures_closure_captures({:?}), ", closure_expr);

        let closure_def_id = self.tcx().hir().local_def_id(closure_expr.hir_id).to_def_id();
        let cl_span = self.tcx().hir().span(closure_expr.hir_id);

        let captures = &self.mc.typeck_results.closure_captures[&closure_def_id];

        for (&var_id, &upvar_id) in captures {
            let upvar_capture = self.mc.typeck_results.upvar_capture(upvar_id);
            let captured_place =
                return_if_err!(self.cat_captured_var(closure_expr.hir_id, cl_span, var_id));
            match upvar_capture {
                ty::UpvarCapture::ByValue(_) => {
                    let mode = copy_or_move(&self.mc, &captured_place);
                    self.delegate.consume(&captured_place, captured_place.hir_id, mode);
                }
                ty::UpvarCapture::ByRef(upvar_borrow) => {
                    self.delegate.borrow(&captured_place, captured_place.hir_id, upvar_borrow.kind);
                }
            }
        }
    }

    /// Handle the case where the current body contains a closure.
    ///
    /// When the current body being handled is a closure, then we must make sure that
    /// - The parent closure only captures Places from the nested closure that are not local to it.
    ///
    /// In the following example the closures `c` only captures `p.x`` even though `incr`
    /// is a capture of the nested closure
    ///
    /// ```rust,ignore(cannot-test-this-because-pseduo-code)
    /// let p = ..;
    /// let c = || {
    ///    let incr = 10;
    ///    let nested = || p.x += incr;
    /// }
    /// ```
    ///
    /// - When reporting the Place back to the Delegate, ensure that the UpvarId uses the enclosing
    /// closure as the DefId.
    fn walk_captures(&mut self, closure_expr: &hir::Expr<'_>) {
        debug!("walk_captures({:?})", closure_expr);

        let closure_def_id = self.tcx().hir().local_def_id(closure_expr.hir_id).to_def_id();
        let upvars = self.tcx().upvars_mentioned(self.body_owner);

        // For purposes of this function, generator and closures are equivalent.
        let body_owner_is_closure = match self.tcx().type_of(self.body_owner.to_def_id()).kind() {
            ty::Closure(..) | ty::Generator(..) => true,
            _ => false,
        };

        if let Some(min_captures) = self.mc.typeck_results.closure_min_captures.get(&closure_def_id)
        {
            for (var_hir_id, min_list) in min_captures.iter() {
                if upvars.map_or(body_owner_is_closure, |upvars| !upvars.contains_key(var_hir_id)) {
                    // The nested closure might be capturing the current (enclosing) closure's local variables.
                    // We check if the root variable is ever mentioned within the enclosing closure, if not
                    // then for the current body (if it's a closure) these aren't captures, we will ignore them.
                    continue;
                }
                for captured_place in min_list {
                    let place = &captured_place.place;
                    let capture_info = captured_place.info;

                    let upvar_id = if body_owner_is_closure {
                        // Mark the place to be captured by the enclosing closure
                        ty::UpvarId::new(*var_hir_id, self.body_owner)
                    } else {
                        ty::UpvarId::new(*var_hir_id, closure_def_id.expect_local())
                    };
                    let place_with_id = PlaceWithHirId::new(
                        capture_info.expr_id.unwrap_or(closure_expr.hir_id),
                        place.base_ty,
                        PlaceBase::Upvar(upvar_id),
                        place.projections.clone(),
                    );

                    match capture_info.capture_kind {
                        ty::UpvarCapture::ByValue(_) => {
                            let mode = copy_or_move(&self.mc, &place_with_id);
                            self.delegate.consume(&place_with_id, place_with_id.hir_id, mode);
                        }
                        ty::UpvarCapture::ByRef(upvar_borrow) => {
                            self.delegate.borrow(
                                &place_with_id,
                                place_with_id.hir_id,
                                upvar_borrow.kind,
                            );
                        }
                    }
                }
            }
        } else if self.mc.typeck_results.closure_captures.contains_key(&closure_def_id) {
            // Handle the case where clippy calls ExprUseVisitor after
            self.walk_captures_closure_captures(closure_expr)
        }
    }

    fn cat_captured_var(
        &mut self,
        closure_hir_id: hir::HirId,
        closure_span: Span,
        var_id: hir::HirId,
    ) -> mc::McResult<PlaceWithHirId<'tcx>> {
        // Create the place for the variable being borrowed, from the
        // perspective of the creator (parent) of the closure.
        let var_ty = self.mc.node_ty(var_id)?;
        self.mc.cat_res(closure_hir_id, closure_span, var_ty, Res::Local(var_id))
    }
}

fn copy_or_move<'a, 'tcx>(
    mc: &mc::MemCategorizationContext<'a, 'tcx>,
    place_with_id: &PlaceWithHirId<'tcx>,
) -> ConsumeMode {
    if !mc.type_is_copy_modulo_regions(
        place_with_id.place.ty(),
        mc.tcx().hir().span(place_with_id.hir_id),
    ) {
        Move
    } else {
        Copy
    }
}
