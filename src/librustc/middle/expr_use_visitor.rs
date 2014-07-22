// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * A different sort of visitor for walking fn bodies.  Unlike the
 * normal visitor, which just walks the entire body in one shot, the
 * `ExprUseVisitor` determines how expressions are being used.
 */

use mc = middle::mem_categorization;
use middle::def;
use middle::freevars;
use middle::pat_util;
use middle::ty;
use middle::typeck::{MethodCall, MethodObject, MethodOrigin, MethodParam};
use middle::typeck::{MethodStatic, MethodStaticUnboxedClosure};
use middle::typeck;
use util::ppaux::Repr;

use std::gc::Gc;
use syntax::ast;
use syntax::codemap::Span;

///////////////////////////////////////////////////////////////////////////
// The Delegate trait

/// This trait defines the callbacks you can expect to receive when
/// employing the ExprUseVisitor.
pub trait Delegate {
    // The value found at `cmt` is either copied or moved, depending
    // on mode.
    fn consume(&mut self,
               consume_id: ast::NodeId,
               consume_span: Span,
               cmt: mc::cmt,
               mode: ConsumeMode);

    // The value found at `cmt` is either copied or moved via the
    // pattern binding `consume_pat`, depending on mode.
    fn consume_pat(&mut self,
                   consume_pat: &ast::Pat,
                   cmt: mc::cmt,
                   mode: ConsumeMode);

    // The value found at `borrow` is being borrowed at the point
    // `borrow_id` for the region `loan_region` with kind `bk`.
    fn borrow(&mut self,
              borrow_id: ast::NodeId,
              borrow_span: Span,
              cmt: mc::cmt,
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
              assignee_cmt: mc::cmt,
              mode: MutateMode);
}

#[deriving(PartialEq)]
pub enum LoanCause {
    ClosureCapture(Span),
    AddrOf,
    AutoRef,
    RefBinding,
    OverloadedOperator,
    ClosureInvocation,
    ForLoop,
}

#[deriving(PartialEq,Show)]
pub enum ConsumeMode {
    Copy,                // reference to x where x has a type that copies
    Move(MoveReason),    // reference to x where x has a type that moves
}

#[deriving(PartialEq,Show)]
pub enum MoveReason {
    DirectRefMove,
    PatBindingMove,
    CaptureMove,
}

#[deriving(PartialEq,Show)]
pub enum MutateMode {
    Init,
    JustWrite,    // x = y
    WriteAndRead, // x += y
}

enum OverloadedCallType {
    FnOverloadedCall,
    FnMutOverloadedCall,
    FnOnceOverloadedCall,
}

impl OverloadedCallType {
    fn from_trait_id(tcx: &ty::ctxt, trait_id: ast::DefId)
                     -> OverloadedCallType {
        for &(maybe_function_trait, overloaded_call_type) in [
            (tcx.lang_items.fn_once_trait(), FnOnceOverloadedCall),
            (tcx.lang_items.fn_mut_trait(), FnMutOverloadedCall),
            (tcx.lang_items.fn_trait(), FnOverloadedCall)
        ].iter() {
            match maybe_function_trait {
                Some(function_trait) if function_trait == trait_id => {
                    return overloaded_call_type
                }
                _ => continue,
            }
        }

        tcx.sess.bug("overloaded call didn't map to known function trait")
    }

    fn from_method_id(tcx: &ty::ctxt, method_id: ast::DefId)
                      -> OverloadedCallType {
        let method_descriptor =
            match tcx.methods.borrow_mut().find(&method_id) {
                None => {
                    tcx.sess.bug("overloaded call method wasn't in method \
                                  map")
                }
                Some(ref method_descriptor) => (*method_descriptor).clone(),
            };
        let impl_id = match method_descriptor.container {
            ty::TraitContainer(_) => {
                tcx.sess.bug("statically resolved overloaded call method \
                              belonged to a trait?!")
            }
            ty::ImplContainer(impl_id) => impl_id,
        };
        let trait_ref = match ty::impl_trait_ref(tcx, impl_id) {
            None => {
                tcx.sess.bug("statically resolved overloaded call impl \
                              didn't implement a trait?!")
            }
            Some(ref trait_ref) => (*trait_ref).clone(),
        };
        OverloadedCallType::from_trait_id(tcx, trait_ref.def_id)
    }

    fn from_method_origin(tcx: &ty::ctxt, origin: &MethodOrigin)
                          -> OverloadedCallType {
        match *origin {
            MethodStatic(def_id) => {
                OverloadedCallType::from_method_id(tcx, def_id)
            }
            MethodStaticUnboxedClosure(def_id) => {
                OverloadedCallType::from_method_id(tcx, def_id)
            }
            MethodParam(ref method_param) => {
                OverloadedCallType::from_trait_id(tcx, method_param.trait_id)
            }
            MethodObject(ref method_object) => {
                OverloadedCallType::from_trait_id(tcx, method_object.trait_id)
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// The ExprUseVisitor type
//
// This is the code that actually walks the tree. Like
// mem_categorization, it requires a TYPER, which is a type that
// supplies types from the tree. After type checking is complete, you
// can just use the tcx as the typer.

pub struct ExprUseVisitor<'d,'t,TYPER> {
    typer: &'t TYPER,
    mc: mc::MemCategorizationContext<'t,TYPER>,
    delegate: &'d mut Delegate,
}

// If the TYPER results in an error, it's because the type check
// failed (or will fail, when the error is uncovered and reported
// during writeback). In this case, we just ignore this part of the
// code.
//
// Note that this macro appears similar to try!(), but, unlike try!(),
// it does not propagate the error.
macro_rules! return_if_err(
    ($inp: expr) => (
        match $inp {
            Ok(v) => v,
            Err(()) => return
        }
    )
)

impl<'d,'t,TYPER:mc::Typer> ExprUseVisitor<'d,'t,TYPER> {
    pub fn new(delegate: &'d mut Delegate,
               typer: &'t TYPER)
               -> ExprUseVisitor<'d,'t,TYPER> {
        ExprUseVisitor { typer: typer,
                         mc: mc::MemCategorizationContext::new(typer),
                         delegate: delegate }
    }

    pub fn walk_fn(&mut self,
                   decl: &ast::FnDecl,
                   body: &ast::Block) {
        self.walk_arg_patterns(decl, body);
        self.walk_block(body);
    }

    fn walk_arg_patterns(&mut self,
                         decl: &ast::FnDecl,
                         body: &ast::Block) {
        for arg in decl.inputs.iter() {
            let arg_ty = ty::node_id_to_type(self.tcx(), arg.pat.id);

            let arg_cmt = self.mc.cat_rvalue(
                arg.id,
                arg.pat.span,
                ty::ReScope(body.id), // Args live only as long as the fn body.
                arg_ty);

            self.walk_pat(arg_cmt, arg.pat.clone());
        }
    }

    fn tcx<'a>(&'a self) -> &'a ty::ctxt {
        self.typer.tcx()
    }

    fn delegate_consume(&mut self,
                        consume_id: ast::NodeId,
                        consume_span: Span,
                        cmt: mc::cmt) {
        let mode = copy_or_move(self.tcx(), cmt.ty, DirectRefMove);
        self.delegate.consume(consume_id, consume_span, cmt, mode);
    }

    fn consume_exprs(&mut self, exprs: &Vec<Gc<ast::Expr>>) {
        for expr in exprs.iter() {
            self.consume_expr(&**expr);
        }
    }

    fn consume_expr(&mut self, expr: &ast::Expr) {
        debug!("consume_expr(expr={})", expr.repr(self.tcx()));

        let cmt = return_if_err!(self.mc.cat_expr(expr));
        self.delegate_consume(expr.id, expr.span, cmt);
        self.walk_expr(expr);
    }

    fn mutate_expr(&mut self,
                   assignment_expr: &ast::Expr,
                   expr: &ast::Expr,
                   mode: MutateMode) {
        let cmt = return_if_err!(self.mc.cat_expr(expr));
        self.delegate.mutate(assignment_expr.id, assignment_expr.span, cmt, mode);
        self.walk_expr(expr);
    }

    fn borrow_expr(&mut self,
                   expr: &ast::Expr,
                   r: ty::Region,
                   bk: ty::BorrowKind,
                   cause: LoanCause) {
        debug!("borrow_expr(expr={}, r={}, bk={})",
               expr.repr(self.tcx()), r.repr(self.tcx()), bk.repr(self.tcx()));

        let cmt = return_if_err!(self.mc.cat_expr(expr));
        self.delegate.borrow(expr.id, expr.span, cmt, r, bk, cause);

        // Note: Unlike consume, we can ignore ExprParen. cat_expr
        // already skips over them, and walk will uncover any
        // attachments or whatever.
        self.walk_expr(expr)
    }

    fn select_from_expr(&mut self, expr: &ast::Expr) {
        self.walk_expr(expr)
    }

    fn walk_expr(&mut self, expr: &ast::Expr) {
        debug!("walk_expr(expr={})", expr.repr(self.tcx()));

        self.walk_adjustment(expr);

        match expr.node {
            ast::ExprParen(ref subexpr) => {
                self.walk_expr(&**subexpr)
            }

            ast::ExprPath(..) => { }

            ast::ExprUnary(ast::UnDeref, ref base) => {      // *base
                if !self.walk_overloaded_operator(expr, &**base, []) {
                    self.select_from_expr(&**base);
                }
            }

            ast::ExprField(ref base, _, _) => {         // base.f
                self.select_from_expr(&**base);
            }

            ast::ExprIndex(ref lhs, ref rhs) => {           // lhs[rhs]
                if !self.walk_overloaded_operator(expr, &**lhs, [rhs.clone()]) {
                    self.select_from_expr(&**lhs);
                    self.consume_expr(&**rhs);
                }
            }

            ast::ExprCall(ref callee, ref args) => {    // callee(args)
                self.walk_callee(expr, &**callee);
                self.consume_exprs(args);
            }

            ast::ExprMethodCall(_, _, ref args) => { // callee.m(args)
                self.consume_exprs(args);
            }

            ast::ExprStruct(_, ref fields, ref opt_with) => {
                self.walk_struct_expr(expr, fields, opt_with.clone());
            }

            ast::ExprTup(ref exprs) => {
                self.consume_exprs(exprs);
            }

            ast::ExprIf(ref cond_expr, ref then_blk, ref opt_else_expr) => {
                self.consume_expr(&**cond_expr);
                self.walk_block(&**then_blk);
                for else_expr in opt_else_expr.iter() {
                    self.consume_expr(&**else_expr);
                }
            }

            ast::ExprMatch(ref discr, ref arms) => {
                // treatment of the discriminant is handled while
                // walking the arms:
                self.walk_expr(&**discr);
                let discr_cmt = return_if_err!(self.mc.cat_expr(&**discr));
                for arm in arms.iter() {
                    self.walk_arm(discr_cmt.clone(), arm);
                }
            }

            ast::ExprVec(ref exprs) => {
                self.consume_exprs(exprs);
            }

            ast::ExprAddrOf(m, ref base) => {   // &base
                // make sure that the thing we are pointing out stays valid
                // for the lifetime `scope_r` of the resulting ptr:
                let expr_ty = ty::expr_ty(self.tcx(), expr);
                if !ty::type_is_bot(expr_ty) {
                    let r = ty::ty_region(self.tcx(), expr.span, expr_ty);
                    let bk = ty::BorrowKind::from_mutbl(m);
                    self.borrow_expr(&**base, r, bk, AddrOf);
                } else {
                    self.walk_expr(&**base);
                }
            }

            ast::ExprInlineAsm(ref ia) => {
                for &(_, ref input) in ia.inputs.iter() {
                    self.consume_expr(&**input);
                }

                for &(_, ref output) in ia.outputs.iter() {
                    self.mutate_expr(expr, &**output, JustWrite);
                }
            }

            ast::ExprBreak(..) |
            ast::ExprAgain(..) |
            ast::ExprLit(..) => {}

            ast::ExprLoop(ref blk, _) => {
                self.walk_block(&**blk);
            }

            ast::ExprWhile(ref cond_expr, ref blk) => {
                self.consume_expr(&**cond_expr);
                self.walk_block(&**blk);
            }

            ast::ExprForLoop(ref pat, ref head, ref blk, _) => {
                // The pattern lives as long as the block.
                debug!("walk_expr for loop case: blk id={}", blk.id);
                self.walk_expr(&**head);

                let head_cmt = return_if_err!(self.mc.cat_expr(&**head));
                self.walk_pat(head_cmt, pat.clone());

                self.walk_block(&**blk);
            }

            ast::ExprUnary(_, ref lhs) => {
                if !self.walk_overloaded_operator(expr, &**lhs, []) {
                    self.consume_expr(&**lhs);
                }
            }

            ast::ExprBinary(_, ref lhs, ref rhs) => {
                if !self.walk_overloaded_operator(expr, &**lhs, [rhs.clone()]) {
                    self.consume_expr(&**lhs);
                    self.consume_expr(&**rhs);
                }
            }

            ast::ExprBlock(ref blk) => {
                self.walk_block(&**blk);
            }

            ast::ExprRet(ref opt_expr) => {
                for expr in opt_expr.iter() {
                    self.consume_expr(&**expr);
                }
            }

            ast::ExprAssign(ref lhs, ref rhs) => {
                self.mutate_expr(expr, &**lhs, JustWrite);
                self.consume_expr(&**rhs);
            }

            ast::ExprCast(ref base, _) => {
                self.consume_expr(&**base);
            }

            ast::ExprAssignOp(_, ref lhs, ref rhs) => {
                // This will have to change if/when we support
                // overloaded operators for `+=` and so forth.
                self.mutate_expr(expr, &**lhs, WriteAndRead);
                self.consume_expr(&**rhs);
            }

            ast::ExprRepeat(ref base, ref count) => {
                self.consume_expr(&**base);
                self.consume_expr(&**count);
            }

            ast::ExprFnBlock(..) |
            ast::ExprUnboxedFn(..) |
            ast::ExprProc(..) => {
                self.walk_captures(expr)
            }

            ast::ExprVstore(ref base, _) => {
                self.consume_expr(&**base);
            }

            ast::ExprBox(ref place, ref base) => {
                self.consume_expr(&**place);
                self.consume_expr(&**base);
            }

            ast::ExprMac(..) => {
                self.tcx().sess.span_bug(
                    expr.span,
                    "macro expression remains after expansion");
            }
        }
    }

    fn walk_callee(&mut self, call: &ast::Expr, callee: &ast::Expr) {
        let callee_ty = ty::expr_ty_adjusted(self.tcx(), callee);
        debug!("walk_callee: callee={} callee_ty={}",
               callee.repr(self.tcx()), callee_ty.repr(self.tcx()));
        match ty::get(callee_ty).sty {
            ty::ty_bare_fn(..) => {
                self.consume_expr(callee);
            }
            ty::ty_closure(ref f) => {
                match f.onceness {
                    ast::Many => {
                        self.borrow_expr(callee,
                                         ty::ReScope(call.id),
                                         ty::UniqueImmBorrow,
                                         ClosureInvocation);
                    }
                    ast::Once => {
                        self.consume_expr(callee);
                    }
                }
            }
            _ => {
                let overloaded_call_type =
                    match self.tcx()
                              .method_map
                              .borrow()
                              .find(&MethodCall::expr(call.id)) {
                    Some(ref method_callee) => {
                        OverloadedCallType::from_method_origin(
                            self.tcx(),
                            &method_callee.origin)
                    }
                    None => {
                        self.tcx().sess.span_bug(
                            callee.span,
                            format!("unexpected callee type {}",
                                    callee_ty.repr(self.tcx())).as_slice())
                    }
                };
                match overloaded_call_type {
                    FnMutOverloadedCall => {
                        self.borrow_expr(callee,
                                         ty::ReScope(call.id),
                                         ty::MutBorrow,
                                         ClosureInvocation);
                    }
                    FnOverloadedCall => {
                        self.borrow_expr(callee,
                                         ty::ReScope(call.id),
                                         ty::ImmBorrow,
                                         ClosureInvocation);
                    }
                    FnOnceOverloadedCall => self.consume_expr(callee),
                }
            }
        }
    }

    fn walk_stmt(&mut self, stmt: &ast::Stmt) {
        match stmt.node {
            ast::StmtDecl(ref decl, _) => {
                match decl.node {
                    ast::DeclLocal(ref local) => {
                        self.walk_local(local.clone());
                    }

                    ast::DeclItem(_) => {
                        // we don't visit nested items in this visitor,
                        // only the fn body we were given.
                    }
                }
            }

            ast::StmtExpr(ref expr, _) |
            ast::StmtSemi(ref expr, _) => {
                self.consume_expr(&**expr);
            }

            ast::StmtMac(..) => {
                self.tcx().sess.span_bug(stmt.span, "unexpanded stmt macro");
            }
        }
    }

    fn walk_local(&mut self, local: Gc<ast::Local>) {
        match local.init {
            None => {
                let delegate = &mut self.delegate;
                pat_util::pat_bindings(&self.typer.tcx().def_map, &*local.pat,
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
                self.walk_pat(init_cmt, local.pat);
            }
        }
    }

    fn walk_block(&mut self, blk: &ast::Block) {
        /*!
         * Indicates that the value of `blk` will be consumed,
         * meaning either copied or moved depending on its type.
         */

        debug!("walk_block(blk.id={:?})", blk.id);

        for stmt in blk.stmts.iter() {
            self.walk_stmt(&**stmt);
        }

        for tail_expr in blk.expr.iter() {
            self.consume_expr(&**tail_expr);
        }
    }

    fn walk_struct_expr(&mut self,
                        _expr: &ast::Expr,
                        fields: &Vec<ast::Field>,
                        opt_with: Option<Gc<ast::Expr>>) {
        // Consume the expressions supplying values for each field.
        for field in fields.iter() {
            self.consume_expr(&*field.expr);
        }

        let with_expr = match opt_with {
            Some(ref w) => { w.clone() }
            None => { return; }
        };

        let with_cmt = return_if_err!(self.mc.cat_expr(&*with_expr));

        // Select just those fields of the `with`
        // expression that will actually be used
        let with_fields = match ty::get(with_cmt.ty).sty {
            ty::ty_struct(did, ref substs) => {
                ty::struct_fields(self.tcx(), did, substs)
            }
            _ => {
                self.tcx().sess.span_bug(
                    with_expr.span,
                    "with expression doesn't evaluate to a struct");
            }
        };

        // Consume those fields of the with expression that are needed.
        for with_field in with_fields.iter() {
            if !contains_field_named(with_field, fields) {
                let cmt_field = self.mc.cat_field(&*with_expr,
                                                  with_cmt.clone(),
                                                  with_field.ident,
                                                  with_field.mt.ty);
                self.delegate_consume(with_expr.id, with_expr.span, cmt_field);
            }
        }

        fn contains_field_named(field: &ty::field,
                                fields: &Vec<ast::Field>)
                                -> bool
        {
            fields.iter().any(
                |f| f.ident.node.name == field.ident.name)
        }
    }

    // Invoke the appropriate delegate calls for anything that gets
    // consumed or borrowed as part of the automatic adjustment
    // process.
    fn walk_adjustment(&mut self, expr: &ast::Expr) {
        let typer = self.typer;
        match typer.adjustments().borrow().find(&expr.id) {
            None => { }
            Some(adjustment) => {
                match *adjustment {
                    ty::AutoAddEnv(..) |
                    ty::AutoObject(..) => {
                        // Creating an object or closure consumes the
                        // input and stores it into the resulting rvalue.
                        debug!("walk_adjustment(AutoAddEnv|AutoObject)");
                        let cmt_unadjusted =
                            return_if_err!(self.mc.cat_expr_unadjusted(expr));
                        self.delegate_consume(expr.id, expr.span, cmt_unadjusted);
                    }
                    ty::AutoDerefRef(ty::AutoDerefRef {
                        autoref: ref opt_autoref,
                        autoderefs: n
                    }) => {
                        self.walk_autoderefs(expr, n);

                        match *opt_autoref {
                            None => { }
                            Some(ref r) => {
                                self.walk_autoref(expr, r, n);
                            }
                        }
                    }
                }
            }
        }
    }

    fn walk_autoderefs(&mut self,
                       expr: &ast::Expr,
                       autoderefs: uint) {
        /*!
         * Autoderefs for overloaded Deref calls in fact reference
         * their receiver. That is, if we have `(*x)` where `x` is of
         * type `Rc<T>`, then this in fact is equivalent to
         * `x.deref()`. Since `deref()` is declared with `&self`, this
         * is an autoref of `x`.
         */
        debug!("walk_autoderefs expr={} autoderefs={}", expr.repr(self.tcx()), autoderefs);

        for i in range(0, autoderefs) {
            let deref_id = typeck::MethodCall::autoderef(expr.id, i);
            match self.typer.node_method_ty(deref_id) {
                None => {}
                Some(method_ty) => {
                    let cmt = return_if_err!(self.mc.cat_expr_autoderefd(expr, i));
                    let self_ty = *ty::ty_fn_args(method_ty).get(0);
                    let (m, r) = match ty::get(self_ty).sty {
                        ty::ty_rptr(r, ref m) => (m.mutbl, r),
                        _ => self.tcx().sess.span_bug(expr.span,
                                format!("bad overloaded deref type {}",
                                    method_ty.repr(self.tcx())).as_slice())
                    };
                    let bk = ty::BorrowKind::from_mutbl(m);
                    self.delegate.borrow(expr.id, expr.span, cmt,
                                         r, bk, AutoRef);
                }
            }
        }
    }

    fn walk_autoref(&mut self,
                    expr: &ast::Expr,
                    autoref: &ty::AutoRef,
                    autoderefs: uint) {
        debug!("walk_autoref expr={} autoderefs={}", expr.repr(self.tcx()), autoderefs);

        let cmt_derefd = return_if_err!(
            self.mc.cat_expr_autoderefd(expr, autoderefs));

        debug!("walk_autoref: cmt_derefd={}", cmt_derefd.repr(self.tcx()));

        match *autoref {
            ty::AutoPtr(r, m) => {
                self.delegate.borrow(expr.id,
                                     expr.span,
                                     cmt_derefd,
                                     r,
                                     ty::BorrowKind::from_mutbl(m),
                                     AutoRef)
            }
            ty::AutoBorrowVec(r, m) | ty::AutoBorrowVecRef(r, m) => {
                let cmt_index = self.mc.cat_index(expr, cmt_derefd, autoderefs+1);
                self.delegate.borrow(expr.id,
                                     expr.span,
                                     cmt_index,
                                     r,
                                     ty::BorrowKind::from_mutbl(m),
                                     AutoRef)
            }
            ty::AutoBorrowObj(r, m) => {
                let cmt_deref = self.mc.cat_deref_obj(expr, cmt_derefd);
                self.delegate.borrow(expr.id,
                                     expr.span,
                                     cmt_deref,
                                     r,
                                     ty::BorrowKind::from_mutbl(m),
                                     AutoRef)
            }
            ty::AutoUnsafe(_) => {}
        }
    }

    fn walk_overloaded_operator(&mut self,
                                expr: &ast::Expr,
                                receiver: &ast::Expr,
                                args: &[Gc<ast::Expr>])
                                -> bool
    {
        if !self.typer.is_method_call(expr.id) {
            return false;
        }

        self.walk_expr(receiver);

        // Arguments (but not receivers) to overloaded operator
        // methods are implicitly autoref'd which sadly does not use
        // adjustments, so we must hardcode the borrow here.

        let r = ty::ReScope(expr.id);
        let bk = ty::ImmBorrow;

        for arg in args.iter() {
            self.borrow_expr(&**arg, r, bk, OverloadedOperator);
        }
        return true;
    }

    fn walk_arm(&mut self, discr_cmt: mc::cmt, arm: &ast::Arm) {
        for &pat in arm.pats.iter() {
            self.walk_pat(discr_cmt.clone(), pat);
        }

        for guard in arm.guard.iter() {
            self.consume_expr(&**guard);
        }

        self.consume_expr(&*arm.body);
    }

    fn walk_pat(&mut self, cmt_discr: mc::cmt, pat: Gc<ast::Pat>) {
        debug!("walk_pat cmt_discr={} pat={}", cmt_discr.repr(self.tcx()),
               pat.repr(self.tcx()));
        let mc = &self.mc;
        let typer = self.typer;
        let tcx = typer.tcx();
        let def_map = &self.typer.tcx().def_map;
        let delegate = &mut self.delegate;
        return_if_err!(mc.cat_pattern(cmt_discr, &*pat, |mc, cmt_pat, pat| {
            if pat_util::pat_is_binding(def_map, pat) {
                let tcx = typer.tcx();

                debug!("binding cmt_pat={} pat={}",
                       cmt_pat.repr(tcx),
                       pat.repr(tcx));

                // pat_ty: the type of the binding being produced.
                let pat_ty = ty::node_id_to_type(tcx, pat.id);

                // Each match binding is effectively an assignment to the
                // binding being produced.
                let def = def_map.borrow().get_copy(&pat.id);
                match mc.cat_def(pat.id, pat.span, pat_ty, def) {
                    Ok(binding_cmt) => {
                        delegate.mutate(pat.id, pat.span, binding_cmt, Init);
                    }
                    Err(_) => { }
                }

                // It is also a borrow or copy/move of the value being matched.
                match pat.node {
                    ast::PatIdent(ast::BindByRef(m), _, _) => {
                        let (r, bk) = {
                            (ty::ty_region(tcx, pat.span, pat_ty),
                             ty::BorrowKind::from_mutbl(m))
                        };
                        delegate.borrow(pat.id, pat.span, cmt_pat,
                                             r, bk, RefBinding);
                    }
                    ast::PatIdent(ast::BindByValue(_), _, _) => {
                        let mode = copy_or_move(typer.tcx(), cmt_pat.ty, PatBindingMove);
                        delegate.consume_pat(pat, cmt_pat, mode);
                    }
                    _ => {
                        typer.tcx().sess.span_bug(
                            pat.span,
                            "binding pattern not an identifier");
                    }
                }
            } else {
                match pat.node {
                    ast::PatVec(_, Some(slice_pat), _) => {
                        // The `slice_pat` here creates a slice into
                        // the original vector.  This is effectively a
                        // borrow of the elements of the vector being
                        // matched.

                        let (slice_cmt, slice_mutbl, slice_r) = {
                            match mc.cat_slice_pattern(cmt_pat, &*slice_pat) {
                                Ok(v) => v,
                                Err(()) => {
                                    tcx.sess.span_bug(slice_pat.span,
                                                      "Err from mc")
                                }
                            }
                        };

                        // Note: We declare here that the borrow
                        // occurs upon entering the `[...]`
                        // pattern. This implies that something like
                        // `[a, ..b]` where `a` is a move is illegal,
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
                        // only work for `~` vectors. It seems simpler
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
    }

    fn walk_captures(&mut self, closure_expr: &ast::Expr) {
        debug!("walk_captures({})", closure_expr.repr(self.tcx()));

        let tcx = self.typer.tcx();
        freevars::with_freevars(tcx, closure_expr.id, |freevars| {
            match freevars::get_capture_mode(self.tcx(), closure_expr.id) {
                freevars::CaptureByRef => {
                    self.walk_by_ref_captures(closure_expr, freevars);
                }
                freevars::CaptureByValue => {
                    self.walk_by_value_captures(closure_expr, freevars);
                }
            }
        });
    }

    fn walk_by_ref_captures(&mut self,
                            closure_expr: &ast::Expr,
                            freevars: &[freevars::freevar_entry]) {
        for freevar in freevars.iter() {
            let id_var = freevar.def.def_id().node;
            let cmt_var = return_if_err!(self.cat_captured_var(closure_expr.id,
                                                               closure_expr.span,
                                                               freevar.def));

            // Lookup the kind of borrow the callee requires, as
            // inferred by regionbk
            let upvar_id = ty::UpvarId { var_id: id_var,
                                         closure_expr_id: closure_expr.id };
            let upvar_borrow = self.tcx().upvar_borrow_map.borrow()
                                   .get_copy(&upvar_id);

            self.delegate.borrow(closure_expr.id,
                                 closure_expr.span,
                                 cmt_var,
                                 upvar_borrow.region,
                                 upvar_borrow.kind,
                                 ClosureCapture(freevar.span));
        }
    }

    fn walk_by_value_captures(&mut self,
                              closure_expr: &ast::Expr,
                              freevars: &[freevars::freevar_entry]) {
        for freevar in freevars.iter() {
            let cmt_var = return_if_err!(self.cat_captured_var(closure_expr.id,
                                                               closure_expr.span,
                                                               freevar.def));
            let mode = copy_or_move(self.tcx(), cmt_var.ty, CaptureMove);
            self.delegate.consume(closure_expr.id, freevar.span, cmt_var, mode);
        }
    }

    fn cat_captured_var(&mut self,
                        closure_id: ast::NodeId,
                        closure_span: Span,
                        upvar_def: def::Def)
                        -> mc::McResult<mc::cmt> {
        // Create the cmt for the variable being borrowed, from the
        // caller's perspective
        let var_id = upvar_def.def_id().node;
        let var_ty = ty::node_id_to_type(self.tcx(), var_id);
        self.mc.cat_def(closure_id, closure_span, var_ty, upvar_def)
    }
}

fn copy_or_move(tcx: &ty::ctxt, ty: ty::t, move_reason: MoveReason) -> ConsumeMode {
    if ty::type_moves_by_default(tcx, ty) { Move(move_reason) } else { Copy }
}

