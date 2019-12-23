//! A different sort of visitor for walking fn bodies. Unlike the
//! normal visitor, which just walks the entire body in one shot, the
//! `ExprUseVisitor` determines how expressions are being used.

pub use self::ConsumeMode::*;
use self::OverloadedCallType::*;

// Export these here so that Clippy can use them.
pub use mc::{Place, PlaceBase, Projection};

use rustc::hir::def::Res;
use rustc::hir::def_id::DefId;
use rustc::hir::ptr::P;
use rustc::hir::{self, PatKind};
use rustc::infer::InferCtxt;
use rustc::ty::{self, adjustment, TyCtxt};

use crate::mem_categorization as mc;
use syntax_pos::Span;

///////////////////////////////////////////////////////////////////////////
// The Delegate trait

/// This trait defines the callbacks you can expect to receive when
/// employing the ExprUseVisitor.
pub trait Delegate<'tcx> {
    // The value found at `place` is either copied or moved, depending
    // on mode.
    fn consume(&mut self, place: &mc::Place<'tcx>, mode: ConsumeMode);

    // The value found at `place` is being borrowed with kind `bk`.
    fn borrow(&mut self, place: &mc::Place<'tcx>, bk: ty::BorrowKind);

    // The path at `place` is being assigned to.
    fn mutate(&mut self, assignee_place: &mc::Place<'tcx>);
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

#[derive(Copy, Clone)]
enum OverloadedCallType {
    FnOverloadedCall,
    FnMutOverloadedCall,
    FnOnceOverloadedCall,
}

impl OverloadedCallType {
    fn from_trait_id(tcx: TyCtxt<'_>, trait_id: DefId) -> OverloadedCallType {
        for &(maybe_function_trait, overloaded_call_type) in &[
            (tcx.lang_items().fn_once_trait(), FnOnceOverloadedCall),
            (tcx.lang_items().fn_mut_trait(), FnMutOverloadedCall),
            (tcx.lang_items().fn_trait(), FnOverloadedCall),
        ] {
            match maybe_function_trait {
                Some(function_trait) if function_trait == trait_id => return overloaded_call_type,
                _ => continue,
            }
        }

        bug!("overloaded call didn't map to known function trait")
    }

    fn from_method_id(tcx: TyCtxt<'_>, method_id: DefId) -> OverloadedCallType {
        let method = tcx.associated_item(method_id);
        OverloadedCallType::from_trait_id(tcx, method.container.id())
    }
}

///////////////////////////////////////////////////////////////////////////
// The ExprUseVisitor type
//
// This is the code that actually walks the tree.
pub struct ExprUseVisitor<'a, 'tcx> {
    mc: mc::MemCategorizationContext<'a, 'tcx>,
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
    /// - `tables` --- typeck results for the code being analyzed
    pub fn new(
        delegate: &'a mut (dyn Delegate<'tcx> + 'a),
        infcx: &'a InferCtxt<'a, 'tcx>,
        body_owner: DefId,
        param_env: ty::ParamEnv<'tcx>,
        tables: &'a ty::TypeckTables<'tcx>,
    ) -> Self {
        ExprUseVisitor {
            mc: mc::MemCategorizationContext::new(infcx, param_env, body_owner, tables),
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

    fn delegate_consume(&mut self, place: &Place<'tcx>) {
        debug!("delegate_consume(place={:?})", place);

        let mode = copy_or_move(&self.mc, place);
        self.delegate.consume(place, mode);
    }

    fn consume_exprs(&mut self, exprs: &[hir::Expr]) {
        for expr in exprs {
            self.consume_expr(&expr);
        }
    }

    pub fn consume_expr(&mut self, expr: &hir::Expr) {
        debug!("consume_expr(expr={:?})", expr);

        let place = return_if_err!(self.mc.cat_expr(expr));
        self.delegate_consume(&place);
        self.walk_expr(expr);
    }

    fn mutate_expr(&mut self, expr: &hir::Expr) {
        let place = return_if_err!(self.mc.cat_expr(expr));
        self.delegate.mutate(&place);
        self.walk_expr(expr);
    }

    fn borrow_expr(&mut self, expr: &hir::Expr, bk: ty::BorrowKind) {
        debug!("borrow_expr(expr={:?}, bk={:?})", expr, bk);

        let place = return_if_err!(self.mc.cat_expr(expr));
        self.delegate.borrow(&place, bk);

        self.walk_expr(expr)
    }

    fn select_from_expr(&mut self, expr: &hir::Expr) {
        self.walk_expr(expr)
    }

    pub fn walk_expr(&mut self, expr: &hir::Expr) {
        debug!("walk_expr(expr={:?})", expr);

        self.walk_adjustment(expr);

        match expr.kind {
            hir::ExprKind::Path(_) => {}

            hir::ExprKind::Type(ref subexpr, _) => self.walk_expr(subexpr),

            hir::ExprKind::Unary(hir::UnDeref, ref base) => {
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
                self.walk_callee(expr, callee);
                self.consume_exprs(args);
            }

            hir::ExprKind::MethodCall(.., ref args) => {
                // callee.m(args)
                self.consume_exprs(args);
            }

            hir::ExprKind::Struct(_, ref fields, ref opt_with) => {
                self.walk_struct_expr(fields, opt_with);
            }

            hir::ExprKind::Tup(ref exprs) => {
                self.consume_exprs(exprs);
            }

            hir::ExprKind::Match(ref discr, ref arms, _) => {
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

            hir::ExprKind::InlineAsm(ref ia) => {
                for (o, output) in ia.inner.outputs.iter().zip(&ia.outputs_exprs) {
                    if o.is_indirect {
                        self.consume_expr(output);
                    } else {
                        self.mutate_expr(output);
                    }
                }
                self.consume_exprs(&ia.inputs_exprs);
            }

            hir::ExprKind::Continue(..) | hir::ExprKind::Lit(..) | hir::ExprKind::Err => {}

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

            hir::ExprKind::Assign(ref lhs, ref rhs) => {
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
                if self.mc.tables.is_method_call(expr) {
                    self.consume_expr(lhs);
                } else {
                    self.mutate_expr(lhs);
                }
                self.consume_expr(rhs);
            }

            hir::ExprKind::Repeat(ref base, _) => {
                self.consume_expr(base);
            }

            hir::ExprKind::Closure(_, _, _, fn_decl_span, _) => {
                self.walk_captures(expr, fn_decl_span);
            }

            hir::ExprKind::Box(ref base) => {
                self.consume_expr(base);
            }

            hir::ExprKind::Yield(ref value, _) => {
                self.consume_expr(value);
            }
        }
    }

    fn walk_callee(&mut self, call: &hir::Expr, callee: &hir::Expr) {
        let callee_ty = return_if_err!(self.mc.expr_ty_adjusted(callee));
        debug!("walk_callee: callee={:?} callee_ty={:?}", callee, callee_ty);
        match callee_ty.kind {
            ty::FnDef(..) | ty::FnPtr(_) => {
                self.consume_expr(callee);
            }
            ty::Error => {}
            _ => {
                if let Some(def_id) = self.mc.tables.type_dependent_def_id(call.hir_id) {
                    match OverloadedCallType::from_method_id(self.tcx(), def_id) {
                        FnMutOverloadedCall => {
                            self.borrow_expr(callee, ty::MutBorrow);
                        }
                        FnOverloadedCall => {
                            self.borrow_expr(callee, ty::ImmBorrow);
                        }
                        FnOnceOverloadedCall => self.consume_expr(callee),
                    }
                } else {
                    self.tcx()
                        .sess
                        .delay_span_bug(call.span, "no type-dependent def for overloaded call");
                }
            }
        }
    }

    fn walk_stmt(&mut self, stmt: &hir::Stmt) {
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

    fn walk_local(&mut self, local: &hir::Local) {
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
    fn walk_block(&mut self, blk: &hir::Block) {
        debug!("walk_block(blk.hir_id={})", blk.hir_id);

        for stmt in &blk.stmts {
            self.walk_stmt(stmt);
        }

        if let Some(ref tail_expr) = blk.expr {
            self.consume_expr(&tail_expr);
        }
    }

    fn walk_struct_expr(&mut self, fields: &[hir::Field], opt_with: &Option<P<hir::Expr>>) {
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
        match with_place.ty.kind {
            ty::Adt(adt, substs) if adt.is_struct() => {
                // Consume those fields of the with expression that are needed.
                for (f_index, with_field) in adt.non_enum_variant().fields.iter().enumerate() {
                    let is_mentioned = fields
                        .iter()
                        .any(|f| self.tcx().field_index(f.hir_id, self.mc.tables) == f_index);
                    if !is_mentioned {
                        let field_place = self.mc.cat_projection(
                            &*with_expr,
                            with_place.clone(),
                            with_field.ty(self.tcx(), substs),
                        );
                        self.delegate_consume(&field_place);
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
    fn walk_adjustment(&mut self, expr: &hir::Expr) {
        let adjustments = self.mc.tables.expr_adjustments(expr);
        let mut place = return_if_err!(self.mc.cat_expr_unadjusted(expr));
        for adjustment in adjustments {
            debug!("walk_adjustment expr={:?} adj={:?}", expr, adjustment);
            match adjustment.kind {
                adjustment::Adjust::NeverToAny | adjustment::Adjust::Pointer(_) => {
                    // Creating a closure/fn-pointer or unsizing consumes
                    // the input and stores it into the resulting rvalue.
                    self.delegate_consume(&place);
                }

                adjustment::Adjust::Deref(None) => {}

                // Autoderefs for overloaded Deref calls in fact reference
                // their receiver. That is, if we have `(*x)` where `x`
                // is of type `Rc<T>`, then this in fact is equivalent to
                // `x.deref()`. Since `deref()` is declared with `&self`,
                // this is an autoref of `x`.
                adjustment::Adjust::Deref(Some(ref deref)) => {
                    let bk = ty::BorrowKind::from_mutbl(deref.mutbl);
                    self.delegate.borrow(&place, bk);
                }

                adjustment::Adjust::Borrow(ref autoref) => {
                    self.walk_autoref(expr, &place, autoref);
                }
            }
            place = return_if_err!(self.mc.cat_expr_adjusted(expr, place, &adjustment));
        }
    }

    /// Walks the autoref `autoref` applied to the autoderef'd
    /// `expr`. `base_place` is the mem-categorized form of `expr`
    /// after all relevant autoderefs have occurred.
    fn walk_autoref(
        &mut self,
        expr: &hir::Expr,
        base_place: &mc::Place<'tcx>,
        autoref: &adjustment::AutoBorrow<'tcx>,
    ) {
        debug!(
            "walk_autoref(expr.hir_id={} base_place={:?} autoref={:?})",
            expr.hir_id, base_place, autoref
        );

        match *autoref {
            adjustment::AutoBorrow::Ref(_, m) => {
                self.delegate.borrow(base_place, ty::BorrowKind::from_mutbl(m.into()));
            }

            adjustment::AutoBorrow::RawPtr(m) => {
                debug!("walk_autoref: expr.hir_id={} base_place={:?}", expr.hir_id, base_place);

                self.delegate.borrow(base_place, ty::BorrowKind::from_mutbl(m));
            }
        }
    }

    fn walk_arm(&mut self, discr_place: &Place<'tcx>, arm: &hir::Arm) {
        self.walk_pat(discr_place, &arm.pat);

        if let Some(hir::Guard::If(ref e)) = arm.guard {
            self.consume_expr(e)
        }

        self.consume_expr(&arm.body);
    }

    /// Walks a pat that occurs in isolation (i.e., top-level of fn argument or
    /// let binding, and *not* a match arm or nested pat.)
    fn walk_irrefutable_pat(&mut self, discr_place: &Place<'tcx>, pat: &hir::Pat) {
        self.walk_pat(discr_place, pat);
    }

    /// The core driver for walking a pattern
    fn walk_pat(&mut self, discr_place: &Place<'tcx>, pat: &hir::Pat) {
        debug!("walk_pat(discr_place={:?}, pat={:?})", discr_place, pat);

        let tcx = self.tcx();
        let ExprUseVisitor { ref mc, ref mut delegate } = *self;
        return_if_err!(mc.cat_pattern(discr_place.clone(), pat, |place, pat| {
            if let PatKind::Binding(_, canonical_id, ..) = pat.kind {
                debug!("walk_pat: binding place={:?} pat={:?}", place, pat,);
                if let Some(&bm) = mc.tables.pat_binding_modes().get(pat.hir_id) {
                    debug!("walk_pat: pat.hir_id={:?} bm={:?}", pat.hir_id, bm);

                    // pat_ty: the type of the binding being produced.
                    let pat_ty = return_if_err!(mc.node_ty(pat.hir_id));
                    debug!("walk_pat: pat_ty={:?}", pat_ty);

                    // Each match binding is effectively an assignment to the
                    // binding being produced.
                    let def = Res::Local(canonical_id);
                    if let Ok(ref binding_place) = mc.cat_res(pat.hir_id, pat.span, pat_ty, def) {
                        delegate.mutate(binding_place);
                    }

                    // It is also a borrow or copy/move of the value being matched.
                    match bm {
                        ty::BindByReference(m) => {
                            let bk = ty::BorrowKind::from_mutbl(m);
                            delegate.borrow(place, bk);
                        }
                        ty::BindByValue(..) => {
                            let mode = copy_or_move(mc, place);
                            debug!("walk_pat binding consuming pat");
                            delegate.consume(place, mode);
                        }
                    }
                } else {
                    tcx.sess.delay_span_bug(pat.span, "missing binding mode");
                }
            }
        }));
    }

    fn walk_captures(&mut self, closure_expr: &hir::Expr, fn_decl_span: Span) {
        debug!("walk_captures({:?})", closure_expr);

        let closure_def_id = self.tcx().hir().local_def_id(closure_expr.hir_id);
        if let Some(upvars) = self.tcx().upvars(closure_def_id) {
            for &var_id in upvars.keys() {
                let upvar_id = ty::UpvarId {
                    var_path: ty::UpvarPath { hir_id: var_id },
                    closure_expr_id: closure_def_id.to_local(),
                };
                let upvar_capture = self.mc.tables.upvar_capture(upvar_id);
                let captured_place = return_if_err!(self.cat_captured_var(
                    closure_expr.hir_id,
                    fn_decl_span,
                    var_id,
                ));
                match upvar_capture {
                    ty::UpvarCapture::ByValue => {
                        let mode = copy_or_move(&self.mc, &captured_place);
                        self.delegate.consume(&captured_place, mode);
                    }
                    ty::UpvarCapture::ByRef(upvar_borrow) => {
                        self.delegate.borrow(&captured_place, upvar_borrow.kind);
                    }
                }
            }
        }
    }

    fn cat_captured_var(
        &mut self,
        closure_hir_id: hir::HirId,
        closure_span: Span,
        var_id: hir::HirId,
    ) -> mc::McResult<mc::Place<'tcx>> {
        // Create the place for the variable being borrowed, from the
        // perspective of the creator (parent) of the closure.
        let var_ty = self.mc.node_ty(var_id)?;
        self.mc.cat_res(closure_hir_id, closure_span, var_ty, Res::Local(var_id))
    }
}

fn copy_or_move<'a, 'tcx>(
    mc: &mc::MemCategorizationContext<'a, 'tcx>,
    place: &Place<'tcx>,
) -> ConsumeMode {
    if !mc.type_is_copy_modulo_regions(place.ty, place.span) { Move } else { Copy }
}
