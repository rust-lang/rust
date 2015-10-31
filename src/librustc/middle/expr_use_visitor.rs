// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A different sort of visitor for walking fn bodies.  Unlike the
//! normal visitor, which just walks the entire body in one shot, the
//! `ExprUseVisitor` determines how expressions are being used.

pub use self::MutateMode::*;
pub use self::LoanCause::*;
pub use self::ConsumeMode::*;
pub use self::MoveReason::*;
pub use self::MatchMode::*;
use self::TrackMatchMode::*;
use self::OverloadedCallType::*;

use middle::{def, pat_util};
use middle::def_id::{DefId};
use middle::infer;
use middle::mem_categorization as mc;
use middle::ty;
use middle::ty::adjustment;

use rustc_front::hir;

use syntax::ast;
use syntax::ptr::P;
use syntax::codemap::Span;

///////////////////////////////////////////////////////////////////////////
// The Delegate trait

/// This trait defines the callbacks you can expect to receive when
/// employing the ExprUseVisitor.
pub trait Delegate<'tcx> {
    // The value found at `cmt` is either copied or moved, depending
    // on mode.
    fn consume(&mut self,
               consume_id: ast::NodeId,
               consume_span: Span,
               cmt: mc::cmt<'tcx>,
               mode: ConsumeMode);

    // The value found at `cmt` has been determined to match the
    // pattern binding `matched_pat`, and its subparts are being
    // copied or moved depending on `mode`.  Note that `matched_pat`
    // is called on all variant/structs in the pattern (i.e., the
    // interior nodes of the pattern's tree structure) while
    // consume_pat is called on the binding identifiers in the pattern
    // (which are leaves of the pattern's tree structure).
    //
    // Note that variants/structs and identifiers are disjoint; thus
    // `matched_pat` and `consume_pat` are never both called on the
    // same input pattern structure (though of `consume_pat` can be
    // called on a subpart of an input passed to `matched_pat).
    fn matched_pat(&mut self,
                   matched_pat: &hir::Pat,
                   cmt: mc::cmt<'tcx>,
                   mode: MatchMode);

    // The value found at `cmt` is either copied or moved via the
    // pattern binding `consume_pat`, depending on mode.
    fn consume_pat(&mut self,
                   consume_pat: &hir::Pat,
                   cmt: mc::cmt<'tcx>,
                   mode: ConsumeMode);

    // The value found at `borrow` is being borrowed at the point
    // `borrow_id` for the region `loan_region` with kind `bk`.
    fn borrow(&mut self,
              borrow_id: ast::NodeId,
              borrow_span: Span,
              cmt: mc::cmt<'tcx>,
              loan_region: ty::Region,
              bk: ty::BorrowKind,
              loan_cause: LoanCause);

    // The local variable `id` is declared but not initialized.
    fn decl_without_init(&mut self,
                         id: ast::NodeId,
                         span: Span);

    // The path at `cmt` is being assigned to.
    fn mutate(&mut self,
              assignment_id: ast::NodeId,
              assignment_span: Span,
              assignee_cmt: mc::cmt<'tcx>,
              mode: MutateMode);
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum LoanCause {
    ClosureCapture(Span),
    AddrOf,
    AutoRef,
    AutoUnsafe,
    RefBinding,
    OverloadedOperator,
    ClosureInvocation,
    ForLoop,
    MatchDiscriminant
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum ConsumeMode {
    Copy,                // reference to x where x has a type that copies
    Move(MoveReason),    // reference to x where x has a type that moves
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum MoveReason {
    DirectRefMove,
    PatBindingMove,
    CaptureMove,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum MatchMode {
    NonBindingMatch,
    BorrowingMatch,
    CopyingMatch,
    MovingMatch,
}

#[derive(Copy, Clone, PartialEq, Debug)]
enum TrackMatchMode {
    Unknown,
    Definite(MatchMode),
    Conflicting,
}

impl TrackMatchMode {
    // Builds up the whole match mode for a pattern from its constituent
    // parts.  The lattice looks like this:
    //
    //          Conflicting
    //            /     \
    //           /       \
    //      Borrowing   Moving
    //           \       /
    //            \     /
    //            Copying
    //               |
    //          NonBinding
    //               |
    //            Unknown
    //
    // examples:
    //
    // * `(_, some_int)` pattern is Copying, since
    //   NonBinding + Copying => Copying
    //
    // * `(some_int, some_box)` pattern is Moving, since
    //   Copying + Moving => Moving
    //
    // * `(ref x, some_box)` pattern is Conflicting, since
    //   Borrowing + Moving => Conflicting
    //
    // Note that the `Unknown` and `Conflicting` states are
    // represented separately from the other more interesting
    // `Definite` states, which simplifies logic here somewhat.
    fn lub(&mut self, mode: MatchMode) {
        *self = match (*self, mode) {
            // Note that clause order below is very significant.
            (Unknown, new) => Definite(new),
            (Definite(old), new) if old == new => Definite(old),

            (Definite(old), NonBindingMatch) => Definite(old),
            (Definite(NonBindingMatch), new) => Definite(new),

            (Definite(old), CopyingMatch) => Definite(old),
            (Definite(CopyingMatch), new) => Definite(new),

            (Definite(_), _) => Conflicting,
            (Conflicting, _) => *self,
        };
    }

    fn match_mode(&self) -> MatchMode {
        match *self {
            Unknown => NonBindingMatch,
            Definite(mode) => mode,
            Conflicting => {
                // Conservatively return MovingMatch to let the
                // compiler continue to make progress.
                MovingMatch
            }
        }
    }
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
    fn from_trait_id(tcx: &ty::ctxt, trait_id: DefId)
                     -> OverloadedCallType {
        for &(maybe_function_trait, overloaded_call_type) in &[
            (tcx.lang_items.fn_once_trait(), FnOnceOverloadedCall),
            (tcx.lang_items.fn_mut_trait(), FnMutOverloadedCall),
            (tcx.lang_items.fn_trait(), FnOverloadedCall)
        ] {
            match maybe_function_trait {
                Some(function_trait) if function_trait == trait_id => {
                    return overloaded_call_type
                }
                _ => continue,
            }
        }

        tcx.sess.bug("overloaded call didn't map to known function trait")
    }

    fn from_method_id(tcx: &ty::ctxt, method_id: DefId)
                      -> OverloadedCallType {
        let method = tcx.impl_or_trait_item(method_id);
        OverloadedCallType::from_trait_id(tcx, method.container().id())
    }
}

///////////////////////////////////////////////////////////////////////////
// The ExprUseVisitor type
//
// This is the code that actually walks the tree. Like
// mem_categorization, it requires a TYPER, which is a type that
// supplies types from the tree. After type checking is complete, you
// can just use the tcx as the typer.
//
// FIXME(stage0): the :'t here is probably only important for stage0
pub struct ExprUseVisitor<'d, 't, 'a: 't, 'tcx:'a+'d> {
    typer: &'t infer::InferCtxt<'a, 'tcx>,
    mc: mc::MemCategorizationContext<'t, 'a, 'tcx>,
    delegate: &'d mut Delegate<'tcx>,
}

// If the TYPER results in an error, it's because the type check
// failed (or will fail, when the error is uncovered and reported
// during writeback). In this case, we just ignore this part of the
// code.
//
// Note that this macro appears similar to try!(), but, unlike try!(),
// it does not propagate the error.
macro_rules! return_if_err {
    ($inp: expr) => (
        match $inp {
            Ok(v) => v,
            Err(()) => {
                debug!("mc reported err");
                return
            }
        }
    )
}

/// Whether the elements of an overloaded operation are passed by value or by reference
enum PassArgs {
    ByValue,
    ByRef,
}

impl<'d,'t,'a,'tcx> ExprUseVisitor<'d,'t,'a,'tcx> {
    pub fn new(delegate: &'d mut (Delegate<'tcx>),
               typer: &'t infer::InferCtxt<'a, 'tcx>)
               -> ExprUseVisitor<'d,'t,'a,'tcx> where 'tcx:'a
    {
        let mc: mc::MemCategorizationContext<'t, 'a, 'tcx> =
            mc::MemCategorizationContext::new(typer);
        ExprUseVisitor { typer: typer, mc: mc, delegate: delegate }
    }

    pub fn walk_fn(&mut self,
                   decl: &hir::FnDecl,
                   body: &hir::Block) {
        self.walk_arg_patterns(decl, body);
        self.walk_block(body);
    }

    fn walk_arg_patterns(&mut self,
                         decl: &hir::FnDecl,
                         body: &hir::Block) {
        for arg in &decl.inputs {
            let arg_ty = return_if_err!(self.typer.node_ty(arg.pat.id));

            let fn_body_scope = self.tcx().region_maps.node_extent(body.id);
            let arg_cmt = self.mc.cat_rvalue(
                arg.id,
                arg.pat.span,
                ty::ReScope(fn_body_scope), // Args live only as long as the fn body.
                arg_ty);

            self.walk_irrefutable_pat(arg_cmt, &*arg.pat);
        }
    }

    fn tcx(&self) -> &'t ty::ctxt<'tcx> {
        self.typer.tcx
    }

    fn delegate_consume(&mut self,
                        consume_id: ast::NodeId,
                        consume_span: Span,
                        cmt: mc::cmt<'tcx>) {
        debug!("delegate_consume(consume_id={}, cmt={:?})",
               consume_id, cmt);

        let mode = copy_or_move(self.typer, &cmt, DirectRefMove);
        self.delegate.consume(consume_id, consume_span, cmt, mode);
    }

    fn consume_exprs(&mut self, exprs: &Vec<P<hir::Expr>>) {
        for expr in exprs {
            self.consume_expr(&**expr);
        }
    }

    pub fn consume_expr(&mut self, expr: &hir::Expr) {
        debug!("consume_expr(expr={:?})", expr);

        let cmt = return_if_err!(self.mc.cat_expr(expr));
        self.delegate_consume(expr.id, expr.span, cmt);
        self.walk_expr(expr);
    }

    fn mutate_expr(&mut self,
                   assignment_expr: &hir::Expr,
                   expr: &hir::Expr,
                   mode: MutateMode) {
        let cmt = return_if_err!(self.mc.cat_expr(expr));
        self.delegate.mutate(assignment_expr.id, assignment_expr.span, cmt, mode);
        self.walk_expr(expr);
    }

    fn borrow_expr(&mut self,
                   expr: &hir::Expr,
                   r: ty::Region,
                   bk: ty::BorrowKind,
                   cause: LoanCause) {
        debug!("borrow_expr(expr={:?}, r={:?}, bk={:?})",
               expr, r, bk);

        let cmt = return_if_err!(self.mc.cat_expr(expr));
        self.delegate.borrow(expr.id, expr.span, cmt, r, bk, cause);

        self.walk_expr(expr)
    }

    fn select_from_expr(&mut self, expr: &hir::Expr) {
        self.walk_expr(expr)
    }

    pub fn walk_expr(&mut self, expr: &hir::Expr) {
        debug!("walk_expr(expr={:?})", expr);

        self.walk_adjustment(expr);

        match expr.node {
            hir::ExprPath(..) => { }

            hir::ExprUnary(hir::UnDeref, ref base) => {      // *base
                if !self.walk_overloaded_operator(expr, &**base, Vec::new(), PassArgs::ByRef) {
                    self.select_from_expr(&**base);
                }
            }

            hir::ExprField(ref base, _) => {         // base.f
                self.select_from_expr(&**base);
            }

            hir::ExprTupField(ref base, _) => {         // base.<n>
                self.select_from_expr(&**base);
            }

            hir::ExprIndex(ref lhs, ref rhs) => {       // lhs[rhs]
                if !self.walk_overloaded_operator(expr,
                                                  &**lhs,
                                                  vec![&**rhs],
                                                  PassArgs::ByValue) {
                    self.select_from_expr(&**lhs);
                    self.consume_expr(&**rhs);
                }
            }

            hir::ExprRange(ref start, ref end) => {
                start.as_ref().map(|e| self.consume_expr(&**e));
                end.as_ref().map(|e| self.consume_expr(&**e));
            }

            hir::ExprCall(ref callee, ref args) => {    // callee(args)
                self.walk_callee(expr, &**callee);
                self.consume_exprs(args);
            }

            hir::ExprMethodCall(_, _, ref args) => { // callee.m(args)
                self.consume_exprs(args);
            }

            hir::ExprStruct(_, ref fields, ref opt_with) => {
                self.walk_struct_expr(expr, fields, opt_with);
            }

            hir::ExprTup(ref exprs) => {
                self.consume_exprs(exprs);
            }

            hir::ExprIf(ref cond_expr, ref then_blk, ref opt_else_expr) => {
                self.consume_expr(&**cond_expr);
                self.walk_block(&**then_blk);
                if let Some(ref else_expr) = *opt_else_expr {
                    self.consume_expr(&**else_expr);
                }
            }

            hir::ExprMatch(ref discr, ref arms, _) => {
                let discr_cmt = return_if_err!(self.mc.cat_expr(&**discr));
                self.borrow_expr(&**discr, ty::ReEmpty, ty::ImmBorrow, MatchDiscriminant);

                // treatment of the discriminant is handled while walking the arms.
                for arm in arms {
                    let mode = self.arm_move_mode(discr_cmt.clone(), arm);
                    let mode = mode.match_mode();
                    self.walk_arm(discr_cmt.clone(), arm, mode);
                }
            }

            hir::ExprVec(ref exprs) => {
                self.consume_exprs(exprs);
            }

            hir::ExprAddrOf(m, ref base) => {   // &base
                // make sure that the thing we are pointing out stays valid
                // for the lifetime `scope_r` of the resulting ptr:
                let expr_ty = return_if_err!(self.typer.node_ty(expr.id));
                if let ty::TyRef(&r, _) = expr_ty.sty {
                    let bk = ty::BorrowKind::from_mutbl(m);
                    self.borrow_expr(&**base, r, bk, AddrOf);
                }
            }

            hir::ExprInlineAsm(ref ia) => {
                for &(_, ref input) in &ia.inputs {
                    self.consume_expr(&**input);
                }

                for &(_, ref output, is_rw) in &ia.outputs {
                    self.mutate_expr(expr, &**output,
                                           if is_rw { WriteAndRead } else { JustWrite });
                }
            }

            hir::ExprBreak(..) |
            hir::ExprAgain(..) |
            hir::ExprLit(..) => {}

            hir::ExprLoop(ref blk, _) => {
                self.walk_block(&**blk);
            }

            hir::ExprWhile(ref cond_expr, ref blk, _) => {
                self.consume_expr(&**cond_expr);
                self.walk_block(&**blk);
            }

            hir::ExprUnary(op, ref lhs) => {
                let pass_args = if ::rustc_front::util::is_by_value_unop(op) {
                    PassArgs::ByValue
                } else {
                    PassArgs::ByRef
                };

                if !self.walk_overloaded_operator(expr, &**lhs, Vec::new(), pass_args) {
                    self.consume_expr(&**lhs);
                }
            }

            hir::ExprBinary(op, ref lhs, ref rhs) => {
                let pass_args = if ::rustc_front::util::is_by_value_binop(op.node) {
                    PassArgs::ByValue
                } else {
                    PassArgs::ByRef
                };

                if !self.walk_overloaded_operator(expr, &**lhs, vec![&**rhs], pass_args) {
                    self.consume_expr(&**lhs);
                    self.consume_expr(&**rhs);
                }
            }

            hir::ExprBlock(ref blk) => {
                self.walk_block(&**blk);
            }

            hir::ExprRet(ref opt_expr) => {
                if let Some(ref expr) = *opt_expr {
                    self.consume_expr(&**expr);
                }
            }

            hir::ExprAssign(ref lhs, ref rhs) => {
                self.mutate_expr(expr, &**lhs, JustWrite);
                self.consume_expr(&**rhs);
            }

            hir::ExprCast(ref base, _) => {
                self.consume_expr(&**base);
            }

            hir::ExprAssignOp(op, ref lhs, ref rhs) => {
                // NB All our assignment operations take the RHS by value
                assert!(::rustc_front::util::is_by_value_binop(op.node));

                if !self.walk_overloaded_operator(expr, lhs, vec![rhs], PassArgs::ByValue) {
                    self.mutate_expr(expr, &**lhs, WriteAndRead);
                    self.consume_expr(&**rhs);
                }
            }

            hir::ExprRepeat(ref base, ref count) => {
                self.consume_expr(&**base);
                self.consume_expr(&**count);
            }

            hir::ExprClosure(..) => {
                self.walk_captures(expr)
            }

            hir::ExprBox(ref base) => {
                self.consume_expr(&**base);
            }
        }
    }

    fn walk_callee(&mut self, call: &hir::Expr, callee: &hir::Expr) {
        let callee_ty = return_if_err!(self.typer.expr_ty_adjusted(callee));
        debug!("walk_callee: callee={:?} callee_ty={:?}",
               callee, callee_ty);
        let call_scope = self.tcx().region_maps.node_extent(call.id);
        match callee_ty.sty {
            ty::TyBareFn(..) => {
                self.consume_expr(callee);
            }
            ty::TyError => { }
            _ => {
                let overloaded_call_type =
                    match self.typer.node_method_id(ty::MethodCall::expr(call.id)) {
                        Some(method_id) => {
                            OverloadedCallType::from_method_id(self.tcx(), method_id)
                        }
                        None => {
                            self.tcx().sess.span_bug(
                                callee.span,
                                &format!("unexpected callee type {}", callee_ty))
                        }
                    };
                match overloaded_call_type {
                    FnMutOverloadedCall => {
                        self.borrow_expr(callee,
                                         ty::ReScope(call_scope),
                                         ty::MutBorrow,
                                         ClosureInvocation);
                    }
                    FnOverloadedCall => {
                        self.borrow_expr(callee,
                                         ty::ReScope(call_scope),
                                         ty::ImmBorrow,
                                         ClosureInvocation);
                    }
                    FnOnceOverloadedCall => self.consume_expr(callee),
                }
            }
        }
    }

    fn walk_stmt(&mut self, stmt: &hir::Stmt) {
        match stmt.node {
            hir::StmtDecl(ref decl, _) => {
                match decl.node {
                    hir::DeclLocal(ref local) => {
                        self.walk_local(&**local);
                    }

                    hir::DeclItem(_) => {
                        // we don't visit nested items in this visitor,
                        // only the fn body we were given.
                    }
                }
            }

            hir::StmtExpr(ref expr, _) |
            hir::StmtSemi(ref expr, _) => {
                self.consume_expr(&**expr);
            }
        }
    }

    fn walk_local(&mut self, local: &hir::Local) {
        match local.init {
            None => {
                let delegate = &mut self.delegate;
                pat_util::pat_bindings(&self.typer.tcx.def_map, &*local.pat,
                                       |_, id, span, _| {
                    delegate.decl_without_init(id, span);
                })
            }

            Some(ref expr) => {
                // Variable declarations with
                // initializers are considered
                // "assigns", which is handled by
                // `walk_pat`:
                self.walk_expr(&**expr);
                let init_cmt = return_if_err!(self.mc.cat_expr(&**expr));
                self.walk_irrefutable_pat(init_cmt, &*local.pat);
            }
        }
    }

    /// Indicates that the value of `blk` will be consumed, meaning either copied or moved
    /// depending on its type.
    fn walk_block(&mut self, blk: &hir::Block) {
        debug!("walk_block(blk.id={})", blk.id);

        for stmt in &blk.stmts {
            self.walk_stmt(&**stmt);
        }

        if let Some(ref tail_expr) = blk.expr {
            self.consume_expr(&**tail_expr);
        }
    }

    fn walk_struct_expr(&mut self,
                        _expr: &hir::Expr,
                        fields: &Vec<hir::Field>,
                        opt_with: &Option<P<hir::Expr>>) {
        // Consume the expressions supplying values for each field.
        for field in fields {
            self.consume_expr(&*field.expr);
        }

        let with_expr = match *opt_with {
            Some(ref w) => &**w,
            None => { return; }
        };

        let with_cmt = return_if_err!(self.mc.cat_expr(&*with_expr));

        // Select just those fields of the `with`
        // expression that will actually be used
        if let ty::TyStruct(def, substs) = with_cmt.ty.sty {
            // Consume those fields of the with expression that are needed.
            for with_field in &def.struct_variant().fields {
                if !contains_field_named(with_field, fields) {
                    let cmt_field = self.mc.cat_field(
                        &*with_expr,
                        with_cmt.clone(),
                        with_field.name,
                        with_field.ty(self.tcx(), substs)
                    );
                    self.delegate_consume(with_expr.id, with_expr.span, cmt_field);
                }
            }
        } else {
            // the base expression should always evaluate to a
            // struct; however, when EUV is run during typeck, it
            // may not. This will generate an error earlier in typeck,
            // so we can just ignore it.
            if !self.tcx().sess.has_errors() {
                self.tcx().sess.span_bug(
                    with_expr.span,
                    "with expression doesn't evaluate to a struct");
            }
        };

        // walk the with expression so that complex expressions
        // are properly handled.
        self.walk_expr(with_expr);

        fn contains_field_named(field: ty::FieldDef,
                                fields: &Vec<hir::Field>)
                                -> bool
        {
            fields.iter().any(
                |f| f.name.node == field.name)
        }
    }

    // Invoke the appropriate delegate calls for anything that gets
    // consumed or borrowed as part of the automatic adjustment
    // process.
    fn walk_adjustment(&mut self, expr: &hir::Expr) {
        let typer = self.typer;
        //NOTE(@jroesch): mixed RefCell borrow causes crash
        let adj = typer.adjustments().get(&expr.id).map(|x| x.clone());
        if let Some(adjustment) = adj {
            match adjustment {
                adjustment::AdjustReifyFnPointer |
                adjustment::AdjustUnsafeFnPointer => {
                    // Creating a closure/fn-pointer or unsizing consumes
                    // the input and stores it into the resulting rvalue.
                    debug!("walk_adjustment(AdjustReifyFnPointer|AdjustUnsafeFnPointer)");
                    let cmt_unadjusted =
                        return_if_err!(self.mc.cat_expr_unadjusted(expr));
                    self.delegate_consume(expr.id, expr.span, cmt_unadjusted);
                }
                adjustment::AdjustDerefRef(ref adj) => {
                    self.walk_autoderefref(expr, adj);
                }
            }
        }
    }

    /// Autoderefs for overloaded Deref calls in fact reference their receiver. That is, if we have
    /// `(*x)` where `x` is of type `Rc<T>`, then this in fact is equivalent to `x.deref()`. Since
    /// `deref()` is declared with `&self`, this is an autoref of `x`.
    fn walk_autoderefs(&mut self,
                       expr: &hir::Expr,
                       autoderefs: usize) {
        debug!("walk_autoderefs expr={:?} autoderefs={}", expr, autoderefs);

        for i in 0..autoderefs {
            let deref_id = ty::MethodCall::autoderef(expr.id, i as u32);
            match self.typer.node_method_ty(deref_id) {
                None => {}
                Some(method_ty) => {
                    let cmt = return_if_err!(self.mc.cat_expr_autoderefd(expr, i));

                    // the method call infrastructure should have
                    // replaced all late-bound regions with variables:
                    let self_ty = method_ty.fn_sig().input(0);
                    let self_ty = self.tcx().no_late_bound_regions(&self_ty).unwrap();

                    let (m, r) = match self_ty.sty {
                        ty::TyRef(r, ref m) => (m.mutbl, r),
                        _ => self.tcx().sess.span_bug(expr.span,
                                &format!("bad overloaded deref type {:?}",
                                    method_ty))
                    };
                    let bk = ty::BorrowKind::from_mutbl(m);
                    self.delegate.borrow(expr.id, expr.span, cmt,
                                         *r, bk, AutoRef);
                }
            }
        }
    }

    fn walk_autoderefref(&mut self,
                         expr: &hir::Expr,
                         adj: &adjustment::AutoDerefRef<'tcx>) {
        debug!("walk_autoderefref expr={:?} adj={:?}",
               expr,
               adj);

        self.walk_autoderefs(expr, adj.autoderefs);

        let cmt_derefd =
            return_if_err!(self.mc.cat_expr_autoderefd(expr, adj.autoderefs));

        let cmt_refd =
            self.walk_autoref(expr, cmt_derefd, adj.autoref);

        if adj.unsize.is_some() {
            // Unsizing consumes the thin pointer and produces a fat one.
            self.delegate_consume(expr.id, expr.span, cmt_refd);
        }
    }


    /// Walks the autoref `opt_autoref` applied to the autoderef'd
    /// `expr`. `cmt_derefd` is the mem-categorized form of `expr`
    /// after all relevant autoderefs have occurred. Because AutoRefs
    /// can be recursive, this function is recursive: it first walks
    /// deeply all the way down the autoref chain, and then processes
    /// the autorefs on the way out. At each point, it returns the
    /// `cmt` for the rvalue that will be produced by introduced an
    /// autoref.
    fn walk_autoref(&mut self,
                    expr: &hir::Expr,
                    cmt_base: mc::cmt<'tcx>,
                    opt_autoref: Option<adjustment::AutoRef<'tcx>>)
                    -> mc::cmt<'tcx>
    {
        debug!("walk_autoref(expr.id={} cmt_derefd={:?} opt_autoref={:?})",
               expr.id,
               cmt_base,
               opt_autoref);

        let cmt_base_ty = cmt_base.ty;

        let autoref = match opt_autoref {
            Some(ref autoref) => autoref,
            None => {
                // No AutoRef.
                return cmt_base;
            }
        };

        match *autoref {
            adjustment::AutoPtr(r, m) => {
                self.delegate.borrow(expr.id,
                                     expr.span,
                                     cmt_base,
                                     *r,
                                     ty::BorrowKind::from_mutbl(m),
                                     AutoRef);
            }

            adjustment::AutoUnsafe(m) => {
                debug!("walk_autoref: expr.id={} cmt_base={:?}",
                       expr.id,
                       cmt_base);

                // Converting from a &T to *T (or &mut T to *mut T) is
                // treated as borrowing it for the enclosing temporary
                // scope.
                let r = ty::ReScope(self.tcx().region_maps.node_extent(expr.id));

                self.delegate.borrow(expr.id,
                                     expr.span,
                                     cmt_base,
                                     r,
                                     ty::BorrowKind::from_mutbl(m),
                                     AutoUnsafe);
            }
        }

        // Construct the categorization for the result of the autoref.
        // This is always an rvalue, since we are producing a new
        // (temporary) indirection.

        let adj_ty = cmt_base_ty.adjust_for_autoref(self.tcx(), opt_autoref);

        self.mc.cat_rvalue_node(expr.id, expr.span, adj_ty)
    }


    // When this returns true, it means that the expression *is* a
    // method-call (i.e. via the operator-overload).  This true result
    // also implies that walk_overloaded_operator already took care of
    // recursively processing the input arguments, and thus the caller
    // should not do so.
    fn walk_overloaded_operator(&mut self,
                                expr: &hir::Expr,
                                receiver: &hir::Expr,
                                rhs: Vec<&hir::Expr>,
                                pass_args: PassArgs)
                                -> bool
    {
        if !self.typer.is_method_call(expr.id) {
            return false;
        }

        match pass_args {
            PassArgs::ByValue => {
                self.consume_expr(receiver);
                for &arg in &rhs {
                    self.consume_expr(arg);
                }

                return true;
            },
            PassArgs::ByRef => {},
        }

        self.walk_expr(receiver);

        // Arguments (but not receivers) to overloaded operator
        // methods are implicitly autoref'd which sadly does not use
        // adjustments, so we must hardcode the borrow here.

        let r = ty::ReScope(self.tcx().region_maps.node_extent(expr.id));
        let bk = ty::ImmBorrow;

        for &arg in &rhs {
            self.borrow_expr(arg, r, bk, OverloadedOperator);
        }
        return true;
    }

    fn arm_move_mode(&mut self, discr_cmt: mc::cmt<'tcx>, arm: &hir::Arm) -> TrackMatchMode {
        let mut mode = Unknown;
        for pat in &arm.pats {
            self.determine_pat_move_mode(discr_cmt.clone(), &**pat, &mut mode);
        }
        mode
    }

    fn walk_arm(&mut self, discr_cmt: mc::cmt<'tcx>, arm: &hir::Arm, mode: MatchMode) {
        for pat in &arm.pats {
            self.walk_pat(discr_cmt.clone(), &**pat, mode);
        }

        if let Some(ref guard) = arm.guard {
            self.consume_expr(&**guard);
        }

        self.consume_expr(&*arm.body);
    }

    /// Walks a pat that occurs in isolation (i.e. top-level of fn
    /// arg or let binding.  *Not* a match arm or nested pat.)
    fn walk_irrefutable_pat(&mut self, cmt_discr: mc::cmt<'tcx>, pat: &hir::Pat) {
        let mut mode = Unknown;
        self.determine_pat_move_mode(cmt_discr.clone(), pat, &mut mode);
        let mode = mode.match_mode();
        self.walk_pat(cmt_discr, pat, mode);
    }

    /// Identifies any bindings within `pat` and accumulates within
    /// `mode` whether the overall pattern/match structure is a move,
    /// copy, or borrow.
    fn determine_pat_move_mode(&mut self,
                               cmt_discr: mc::cmt<'tcx>,
                               pat: &hir::Pat,
                               mode: &mut TrackMatchMode) {
        debug!("determine_pat_move_mode cmt_discr={:?} pat={:?}", cmt_discr,
               pat);
        return_if_err!(self.mc.cat_pattern(cmt_discr, pat, |_mc, cmt_pat, pat| {
            let tcx = self.tcx();
            let def_map = &self.tcx().def_map;
            if pat_util::pat_is_binding(def_map, pat) {
                match pat.node {
                    hir::PatIdent(hir::BindByRef(_), _, _) =>
                        mode.lub(BorrowingMatch),
                    hir::PatIdent(hir::BindByValue(_), _, _) => {
                        match copy_or_move(self.typer, &cmt_pat, PatBindingMove) {
                            Copy => mode.lub(CopyingMatch),
                            Move(_) => mode.lub(MovingMatch),
                        }
                    }
                    _ => {
                        tcx.sess.span_bug(
                            pat.span,
                            "binding pattern not an identifier");
                    }
                }
            }
        }));
    }

    /// The core driver for walking a pattern; `match_mode` must be
    /// established up front, e.g. via `determine_pat_move_mode` (see
    /// also `walk_irrefutable_pat` for patterns that stand alone).
    fn walk_pat(&mut self,
                cmt_discr: mc::cmt<'tcx>,
                pat: &hir::Pat,
                match_mode: MatchMode) {
        debug!("walk_pat cmt_discr={:?} pat={:?}", cmt_discr,
               pat);

        let mc = &self.mc;
        let typer = self.typer;
        let def_map = &self.tcx().def_map;
        let delegate = &mut self.delegate;
        return_if_err!(mc.cat_pattern(cmt_discr.clone(), pat, |mc, cmt_pat, pat| {
            if pat_util::pat_is_binding(def_map, pat) {
                let tcx = typer.tcx;

                debug!("binding cmt_pat={:?} pat={:?} match_mode={:?}",
                       cmt_pat,
                       pat,
                       match_mode);

                // pat_ty: the type of the binding being produced.
                let pat_ty = return_if_err!(typer.node_ty(pat.id));

                // Each match binding is effectively an assignment to the
                // binding being produced.
                let def = def_map.borrow().get(&pat.id).unwrap().full_def();
                match mc.cat_def(pat.id, pat.span, pat_ty, def) {
                    Ok(binding_cmt) => {
                        delegate.mutate(pat.id, pat.span, binding_cmt, Init);
                    }
                    Err(_) => { }
                }

                // It is also a borrow or copy/move of the value being matched.
                match pat.node {
                    hir::PatIdent(hir::BindByRef(m), _, _) => {
                        if let ty::TyRef(&r, _) = pat_ty.sty {
                            let bk = ty::BorrowKind::from_mutbl(m);
                            delegate.borrow(pat.id, pat.span, cmt_pat,
                                            r, bk, RefBinding);
                        }
                    }
                    hir::PatIdent(hir::BindByValue(_), _, _) => {
                        let mode = copy_or_move(typer, &cmt_pat, PatBindingMove);
                        debug!("walk_pat binding consuming pat");
                        delegate.consume_pat(pat, cmt_pat, mode);
                    }
                    _ => {
                        tcx.sess.span_bug(
                            pat.span,
                            "binding pattern not an identifier");
                    }
                }
            } else {
                match pat.node {
                    hir::PatVec(_, Some(ref slice_pat), _) => {
                        // The `slice_pat` here creates a slice into
                        // the original vector.  This is effectively a
                        // borrow of the elements of the vector being
                        // matched.

                        let (slice_cmt, slice_mutbl, slice_r) =
                            return_if_err!(mc.cat_slice_pattern(cmt_pat, &**slice_pat));

                        // Note: We declare here that the borrow
                        // occurs upon entering the `[...]`
                        // pattern. This implies that something like
                        // `[a; b]` where `a` is a move is illegal,
                        // because the borrow is already in effect.
                        // In fact such a move would be safe-ish, but
                        // it effectively *requires* that we use the
                        // nulling out semantics to indicate when a
                        // value has been moved, which we are trying
                        // to move away from.  Otherwise, how can we
                        // indicate that the first element in the
                        // vector has been moved?  Eventually, we
                        // could perhaps modify this rule to permit
                        // `[..a, b]` where `b` is a move, because in
                        // that case we can adjust the length of the
                        // original vec accordingly, but we'd have to
                        // make trans do the right thing, and it would
                        // only work for `Box<[T]>`s. It seems simpler
                        // to just require that people call
                        // `vec.pop()` or `vec.unshift()`.
                        let slice_bk = ty::BorrowKind::from_mutbl(slice_mutbl);
                        delegate.borrow(pat.id, pat.span,
                                        slice_cmt, slice_r,
                                        slice_bk, RefBinding);
                    }
                    _ => { }
                }
            }
        }));

        // Do a second pass over the pattern, calling `matched_pat` on
        // the interior nodes (enum variants and structs), as opposed
        // to the above loop's visit of than the bindings that form
        // the leaves of the pattern tree structure.
        return_if_err!(mc.cat_pattern(cmt_discr, pat, |mc, cmt_pat, pat| {
            let def_map = def_map.borrow();
            let tcx = typer.tcx;

            match pat.node {
                hir::PatEnum(_, _) | hir::PatQPath(..) |
                hir::PatIdent(_, _, None) | hir::PatStruct(..) => {
                    match def_map.get(&pat.id).map(|d| d.full_def()) {
                        None => {
                            // no definition found: pat is not a
                            // struct or enum pattern.
                        }

                        Some(def::DefVariant(enum_did, variant_did, _is_struct)) => {
                            let downcast_cmt =
                                if tcx.lookup_adt_def(enum_did).is_univariant() {
                                    cmt_pat
                                } else {
                                    let cmt_pat_ty = cmt_pat.ty;
                                    mc.cat_downcast(pat, cmt_pat, cmt_pat_ty, variant_did)
                                };

                            debug!("variant downcast_cmt={:?} pat={:?}",
                                   downcast_cmt,
                                   pat);

                            delegate.matched_pat(pat, downcast_cmt, match_mode);
                        }

                        Some(def::DefStruct(..)) | Some(def::DefTy(_, false)) => {
                            // A struct (in either the value or type
                            // namespace; we encounter the former on
                            // e.g. patterns for unit structs).

                            debug!("struct cmt_pat={:?} pat={:?}",
                                   cmt_pat,
                                   pat);

                            delegate.matched_pat(pat, cmt_pat, match_mode);
                        }

                        Some(def::DefConst(..)) |
                        Some(def::DefAssociatedConst(..)) |
                        Some(def::DefLocal(..)) => {
                            // This is a leaf (i.e. identifier binding
                            // or constant value to match); thus no
                            // `matched_pat` call.
                        }

                        Some(def @ def::DefTy(_, true)) => {
                            // An enum's type -- should never be in a
                            // pattern.

                            if !tcx.sess.has_errors() {
                                let msg = format!("Pattern has unexpected type: {:?} and type {:?}",
                                                  def,
                                                  cmt_pat.ty);
                                tcx.sess.span_bug(pat.span, &msg)
                            }
                        }

                        Some(def) => {
                            // Remaining cases are e.g. DefFn, to
                            // which identifiers within patterns
                            // should not resolve. However, we do
                            // encouter this when using the
                            // expr-use-visitor during typeck. So just
                            // ignore it, an error should have been
                            // reported.

                            if !tcx.sess.has_errors() {
                                let msg = format!("Pattern has unexpected def: {:?} and type {:?}",
                                                  def,
                                                  cmt_pat.ty);
                                tcx.sess.span_bug(pat.span, &msg[..])
                            }
                        }
                    }
                }

                hir::PatIdent(_, _, Some(_)) => {
                    // Do nothing; this is a binding (not an enum
                    // variant or struct), and the cat_pattern call
                    // will visit the substructure recursively.
                }

                hir::PatWild(_) | hir::PatTup(..) | hir::PatBox(..) |
                hir::PatRegion(..) | hir::PatLit(..) | hir::PatRange(..) |
                hir::PatVec(..) => {
                    // Similarly, each of these cases does not
                    // correspond to an enum variant or struct, so we
                    // do not do any `matched_pat` calls for these
                    // cases either.
                }
            }
        }));
    }

    fn walk_captures(&mut self, closure_expr: &hir::Expr) {
        debug!("walk_captures({:?})", closure_expr);

        self.tcx().with_freevars(closure_expr.id, |freevars| {
            for freevar in freevars {
                let id_var = freevar.def.var_id();
                let upvar_id = ty::UpvarId { var_id: id_var,
                                             closure_expr_id: closure_expr.id };
                let upvar_capture = self.typer.upvar_capture(upvar_id).unwrap();
                let cmt_var = return_if_err!(self.cat_captured_var(closure_expr.id,
                                                                   closure_expr.span,
                                                                   freevar.def));
                match upvar_capture {
                    ty::UpvarCapture::ByValue => {
                        let mode = copy_or_move(self.typer, &cmt_var, CaptureMove);
                        self.delegate.consume(closure_expr.id, freevar.span, cmt_var, mode);
                    }
                    ty::UpvarCapture::ByRef(upvar_borrow) => {
                        self.delegate.borrow(closure_expr.id,
                                             closure_expr.span,
                                             cmt_var,
                                             upvar_borrow.region,
                                             upvar_borrow.kind,
                                             ClosureCapture(freevar.span));
                    }
                }
            }
        });
    }

    fn cat_captured_var(&mut self,
                        closure_id: ast::NodeId,
                        closure_span: Span,
                        upvar_def: def::Def)
                        -> mc::McResult<mc::cmt<'tcx>> {
        // Create the cmt for the variable being borrowed, from the
        // caller's perspective
        let var_id = upvar_def.var_id();
        let var_ty = try!(self.typer.node_ty(var_id));
        self.mc.cat_def(closure_id, closure_span, var_ty, upvar_def)
    }
}

fn copy_or_move<'a, 'tcx>(typer: &infer::InferCtxt<'a, 'tcx>,
                      cmt: &mc::cmt<'tcx>,
                      move_reason: MoveReason)
                      -> ConsumeMode
{
    if typer.type_moves_by_default(cmt.ty, cmt.span) {
        Move(move_reason)
    } else {
        Copy
    }
}
