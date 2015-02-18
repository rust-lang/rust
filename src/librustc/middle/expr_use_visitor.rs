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

use middle::{def, region, pat_util};
use middle::mem_categorization as mc;
use middle::mem_categorization::Typer;
use middle::ty::{self};
use middle::ty::{MethodCall, MethodObject, MethodTraitObject};
use middle::ty::{MethodOrigin, MethodParam, MethodTypeParam};
use middle::ty::{MethodStatic, MethodStaticClosure};
use util::ppaux::Repr;

use syntax::{ast, ast_util};
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
                   matched_pat: &ast::Pat,
                   cmt: mc::cmt<'tcx>,
                   mode: MatchMode);

    // The value found at `cmt` is either copied or moved via the
    // pattern binding `consume_pat`, depending on mode.
    fn consume_pat(&mut self,
                   consume_pat: &ast::Pat,
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

#[derive(Copy, PartialEq, Debug)]
pub enum LoanCause {
    ClosureCapture(Span),
    AddrOf,
    AutoRef,
    RefBinding,
    OverloadedOperator,
    ClosureInvocation,
    ForLoop,
    MatchDiscriminant
}

#[derive(Copy, PartialEq, Debug)]
pub enum ConsumeMode {
    Copy,                // reference to x where x has a type that copies
    Move(MoveReason),    // reference to x where x has a type that moves
}

#[derive(Copy, PartialEq, Debug)]
pub enum MoveReason {
    DirectRefMove,
    PatBindingMove,
    CaptureMove,
}

#[derive(Copy, PartialEq, Debug)]
pub enum MatchMode {
    NonBindingMatch,
    BorrowingMatch,
    CopyingMatch,
    MovingMatch,
}

#[derive(Copy, PartialEq, Debug)]
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

#[derive(Copy, PartialEq, Debug)]
pub enum MutateMode {
    Init,
    JustWrite,    // x = y
    WriteAndRead, // x += y
}

#[derive(Copy)]
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
        let method_descriptor = match ty::impl_or_trait_item(tcx, method_id) {
            ty::MethodTraitItem(ref method_descriptor) => {
                (*method_descriptor).clone()
            }
            ty::TypeTraitItem(_) => {
                tcx.sess.bug("overloaded call method wasn't in method map")
            }
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

    fn from_closure(tcx: &ty::ctxt, closure_did: ast::DefId)
                    -> OverloadedCallType {
        let trait_did =
            tcx.closure_kinds
               .borrow()
               .get(&closure_did)
               .expect("OverloadedCallType::from_closure: didn't find closure id")
               .trait_did(tcx);
        OverloadedCallType::from_trait_id(tcx, trait_did)
    }

    fn from_method_origin(tcx: &ty::ctxt, origin: &MethodOrigin)
                          -> OverloadedCallType {
        match *origin {
            MethodStatic(def_id) => {
                OverloadedCallType::from_method_id(tcx, def_id)
            }
            MethodStaticClosure(def_id) => {
                OverloadedCallType::from_closure(tcx, def_id)
            }
            MethodTypeParam(MethodParam { ref trait_ref, .. }) |
            MethodTraitObject(MethodObject { ref trait_ref, .. }) => {
                OverloadedCallType::from_trait_id(tcx, trait_ref.def_id)
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

pub struct ExprUseVisitor<'d,'t,'tcx:'t,TYPER:'t> {
    typer: &'t TYPER,
    mc: mc::MemCategorizationContext<'t,TYPER>,
    delegate: &'d mut (Delegate<'tcx>+'d),
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
            Err(()) => return
        }
    )
}

/// Whether the elements of an overloaded operation are passed by value or by reference
enum PassArgs {
    ByValue,
    ByRef,
}

impl<'d,'t,'tcx,TYPER:mc::Typer<'tcx>> ExprUseVisitor<'d,'t,'tcx,TYPER> {
    pub fn new(delegate: &'d mut Delegate<'tcx>,
               typer: &'t TYPER)
               -> ExprUseVisitor<'d,'t,'tcx,TYPER> {
        ExprUseVisitor {
            typer: typer,
            mc: mc::MemCategorizationContext::new(typer),
            delegate: delegate,
        }
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
        for arg in &decl.inputs {
            let arg_ty = return_if_err!(self.typer.node_ty(arg.pat.id));

            let fn_body_scope = region::CodeExtent::from_node_id(body.id);
            let arg_cmt = self.mc.cat_rvalue(
                arg.id,
                arg.pat.span,
                ty::ReScope(fn_body_scope), // Args live only as long as the fn body.
                arg_ty);

            self.walk_irrefutable_pat(arg_cmt, &*arg.pat);
        }
    }

    fn tcx(&self) -> &'t ty::ctxt<'tcx> {
        self.typer.tcx()
    }

    fn delegate_consume(&mut self,
                        consume_id: ast::NodeId,
                        consume_span: Span,
                        cmt: mc::cmt<'tcx>) {
        debug!("delegate_consume(consume_id={}, cmt={})",
               consume_id, cmt.repr(self.tcx()));

        let mode = copy_or_move(self.typer, &cmt, DirectRefMove);
        self.delegate.consume(consume_id, consume_span, cmt, mode);
    }

    fn consume_exprs(&mut self, exprs: &Vec<P<ast::Expr>>) {
        for expr in exprs {
            self.consume_expr(&**expr);
        }
    }

    pub fn consume_expr(&mut self, expr: &ast::Expr) {
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

    pub fn walk_expr(&mut self, expr: &ast::Expr) {
        debug!("walk_expr(expr={})", expr.repr(self.tcx()));

        self.walk_adjustment(expr);

        match expr.node {
            ast::ExprParen(ref subexpr) => {
                self.walk_expr(&**subexpr)
            }

            ast::ExprPath(_) | ast::ExprQPath(_) => { }

            ast::ExprUnary(ast::UnDeref, ref base) => {      // *base
                if !self.walk_overloaded_operator(expr, &**base, Vec::new(), PassArgs::ByRef) {
                    self.select_from_expr(&**base);
                }
            }

            ast::ExprField(ref base, _) => {         // base.f
                self.select_from_expr(&**base);
            }

            ast::ExprTupField(ref base, _) => {         // base.<n>
                self.select_from_expr(&**base);
            }

            ast::ExprIndex(ref lhs, ref rhs) => {       // lhs[rhs]
                if !self.walk_overloaded_operator(expr,
                                                  &**lhs,
                                                  vec![&**rhs],
                                                  PassArgs::ByRef) {
                    self.select_from_expr(&**lhs);
                    self.consume_expr(&**rhs);
                }
            }

            ast::ExprRange(ref start, ref end) => {
                start.as_ref().map(|e| self.consume_expr(&**e));
                end.as_ref().map(|e| self.consume_expr(&**e));
            }

            ast::ExprCall(ref callee, ref args) => {    // callee(args)
                self.walk_callee(expr, &**callee);
                self.consume_exprs(args);
            }

            ast::ExprMethodCall(_, _, ref args) => { // callee.m(args)
                self.consume_exprs(args);
            }

            ast::ExprStruct(_, ref fields, ref opt_with) => {
                self.walk_struct_expr(expr, fields, opt_with);
            }

            ast::ExprTup(ref exprs) => {
                self.consume_exprs(exprs);
            }

            ast::ExprIf(ref cond_expr, ref then_blk, ref opt_else_expr) => {
                self.consume_expr(&**cond_expr);
                self.walk_block(&**then_blk);
                if let Some(ref else_expr) = *opt_else_expr {
                    self.consume_expr(&**else_expr);
                }
            }

            ast::ExprIfLet(..) => {
                self.tcx().sess.span_bug(expr.span, "non-desugared ExprIfLet");
            }

            ast::ExprMatch(ref discr, ref arms, _) => {
                let discr_cmt = return_if_err!(self.mc.cat_expr(&**discr));
                self.borrow_expr(&**discr, ty::ReEmpty, ty::ImmBorrow, MatchDiscriminant);

                // treatment of the discriminant is handled while walking the arms.
                for arm in arms {
                    let mode = self.arm_move_mode(discr_cmt.clone(), arm);
                    let mode = mode.match_mode();
                    self.walk_arm(discr_cmt.clone(), arm, mode);
                }
            }

            ast::ExprVec(ref exprs) => {
                self.consume_exprs(exprs);
            }

            ast::ExprAddrOf(m, ref base) => {   // &base
                // make sure that the thing we are pointing out stays valid
                // for the lifetime `scope_r` of the resulting ptr:
                let expr_ty = return_if_err!(self.typer.node_ty(expr.id));
                let r = ty::ty_region(self.tcx(), expr.span, expr_ty);
                let bk = ty::BorrowKind::from_mutbl(m);
                self.borrow_expr(&**base, r, bk, AddrOf);
            }

            ast::ExprInlineAsm(ref ia) => {
                for &(_, ref input) in &ia.inputs {
                    self.consume_expr(&**input);
                }

                for &(_, ref output, is_rw) in &ia.outputs {
                    self.mutate_expr(expr, &**output,
                                           if is_rw { WriteAndRead } else { JustWrite });
                }
            }

            ast::ExprBreak(..) |
            ast::ExprAgain(..) |
            ast::ExprLit(..) => {}

            ast::ExprLoop(ref blk, _) => {
                self.walk_block(&**blk);
            }

            ast::ExprWhile(ref cond_expr, ref blk, _) => {
                self.consume_expr(&**cond_expr);
                self.walk_block(&**blk);
            }

            ast::ExprWhileLet(..) => {
                self.tcx().sess.span_bug(expr.span, "non-desugared ExprWhileLet");
            }

            ast::ExprForLoop(..) => {
                self.tcx().sess.span_bug(expr.span, "non-desugared ExprForLoop");
            }

            ast::ExprUnary(op, ref lhs) => {
                let pass_args = if ast_util::is_by_value_unop(op) {
                    PassArgs::ByValue
                } else {
                    PassArgs::ByRef
                };

                if !self.walk_overloaded_operator(expr, &**lhs, Vec::new(), pass_args) {
                    self.consume_expr(&**lhs);
                }
            }

            ast::ExprBinary(op, ref lhs, ref rhs) => {
                let pass_args = if ast_util::is_by_value_binop(op.node) {
                    PassArgs::ByValue
                } else {
                    PassArgs::ByRef
                };

                if !self.walk_overloaded_operator(expr, &**lhs, vec![&**rhs], pass_args) {
                    self.consume_expr(&**lhs);
                    self.consume_expr(&**rhs);
                }
            }

            ast::ExprBlock(ref blk) => {
                self.walk_block(&**blk);
            }

            ast::ExprRet(ref opt_expr) => {
                if let Some(ref expr) = *opt_expr {
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

            ast::ExprClosure(..) => {
                self.walk_captures(expr)
            }

            ast::ExprBox(ref place, ref base) => {
                match *place {
                    Some(ref place) => self.consume_expr(&**place),
                    None => {}
                }
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
        let callee_ty = return_if_err!(self.typer.expr_ty_adjusted(callee));
        debug!("walk_callee: callee={} callee_ty={}",
               callee.repr(self.tcx()), callee_ty.repr(self.tcx()));
        let call_scope = region::CodeExtent::from_node_id(call.id);
        match callee_ty.sty {
            ty::ty_bare_fn(..) => {
                self.consume_expr(callee);
            }
            ty::ty_err => { }
            _ => {
                let overloaded_call_type =
                    match self.typer.node_method_origin(MethodCall::expr(call.id)) {
                        Some(method_origin) => {
                            OverloadedCallType::from_method_origin(
                                self.tcx(),
                                &method_origin)
                        }
                        None => {
                            self.tcx().sess.span_bug(
                                callee.span,
                                &format!("unexpected callee type {}", callee_ty.repr(self.tcx())))
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

    fn walk_stmt(&mut self, stmt: &ast::Stmt) {
        match stmt.node {
            ast::StmtDecl(ref decl, _) => {
                match decl.node {
                    ast::DeclLocal(ref local) => {
                        self.walk_local(&**local);
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

    fn walk_local(&mut self, local: &ast::Local) {
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
                self.walk_irrefutable_pat(init_cmt, &*local.pat);
            }
        }
    }

    /// Indicates that the value of `blk` will be consumed, meaning either copied or moved
    /// depending on its type.
    fn walk_block(&mut self, blk: &ast::Block) {
        debug!("walk_block(blk.id={})", blk.id);

        for stmt in &blk.stmts {
            self.walk_stmt(&**stmt);
        }

        if let Some(ref tail_expr) = blk.expr {
            self.consume_expr(&**tail_expr);
        }
    }

    fn walk_struct_expr(&mut self,
                        _expr: &ast::Expr,
                        fields: &Vec<ast::Field>,
                        opt_with: &Option<P<ast::Expr>>) {
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
        let with_fields = match with_cmt.ty.sty {
            ty::ty_struct(did, substs) => {
                ty::struct_fields(self.tcx(), did, substs)
            }
            _ => {
                // the base expression should always evaluate to a
                // struct; however, when EUV is run during typeck, it
                // may not. This will generate an error earlier in typeck,
                // so we can just ignore it.
                if !self.tcx().sess.has_errors() {
                    self.tcx().sess.span_bug(
                        with_expr.span,
                        "with expression doesn't evaluate to a struct");
                }
                assert!(self.tcx().sess.has_errors());
                vec!()
            }
        };

        // Consume those fields of the with expression that are needed.
        for with_field in &with_fields {
            if !contains_field_named(with_field, fields) {
                let cmt_field = self.mc.cat_field(&*with_expr,
                                                  with_cmt.clone(),
                                                  with_field.name,
                                                  with_field.mt.ty);
                self.delegate_consume(with_expr.id, with_expr.span, cmt_field);
            }
        }

        // walk the with expression so that complex expressions
        // are properly handled.
        self.walk_expr(with_expr);

        fn contains_field_named(field: &ty::field,
                                fields: &Vec<ast::Field>)
                                -> bool
        {
            fields.iter().any(
                |f| f.ident.node.name == field.name)
        }
    }

    // Invoke the appropriate delegate calls for anything that gets
    // consumed or borrowed as part of the automatic adjustment
    // process.
    fn walk_adjustment(&mut self, expr: &ast::Expr) {
        let typer = self.typer;
        match typer.adjustments().borrow().get(&expr.id) {
            None => { }
            Some(adjustment) => {
                match *adjustment {
                    ty::AdjustReifyFnPointer(..) => {
                        // Creating a closure/fn-pointer consumes the
                        // input and stores it into the resulting
                        // rvalue.
                        debug!("walk_adjustment(AutoAddEnv|AdjustReifyFnPointer)");
                        let cmt_unadjusted =
                            return_if_err!(self.mc.cat_expr_unadjusted(expr));
                        self.delegate_consume(expr.id, expr.span, cmt_unadjusted);
                    }
                    ty::AdjustDerefRef(ty::AutoDerefRef {
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

    /// Autoderefs for overloaded Deref calls in fact reference their receiver. That is, if we have
    /// `(*x)` where `x` is of type `Rc<T>`, then this in fact is equivalent to `x.deref()`. Since
    /// `deref()` is declared with `&self`, this is an autoref of `x`.
    fn walk_autoderefs(&mut self,
                       expr: &ast::Expr,
                       autoderefs: uint) {
        debug!("walk_autoderefs expr={} autoderefs={}", expr.repr(self.tcx()), autoderefs);

        for i in 0..autoderefs {
            let deref_id = ty::MethodCall::autoderef(expr.id, i);
            match self.typer.node_method_ty(deref_id) {
                None => {}
                Some(method_ty) => {
                    let cmt = return_if_err!(self.mc.cat_expr_autoderefd(expr, i));

                    // the method call infrastructure should have
                    // replaced all late-bound regions with variables:
                    let self_ty = ty::ty_fn_sig(method_ty).input(0);
                    let self_ty = ty::no_late_bound_regions(self.tcx(), &self_ty).unwrap();

                    let (m, r) = match self_ty.sty {
                        ty::ty_rptr(r, ref m) => (m.mutbl, r),
                        _ => self.tcx().sess.span_bug(expr.span,
                                &format!("bad overloaded deref type {}",
                                    method_ty.repr(self.tcx()))[])
                    };
                    let bk = ty::BorrowKind::from_mutbl(m);
                    self.delegate.borrow(expr.id, expr.span, cmt,
                                         *r, bk, AutoRef);
                }
            }
        }
    }

    fn walk_autoref(&mut self,
                    expr: &ast::Expr,
                    autoref: &ty::AutoRef,
                    n: uint) {
        debug!("walk_autoref expr={}", expr.repr(self.tcx()));

        // Match for unique trait coercions first, since we don't need the
        // call to cat_expr_autoderefd.
        match *autoref {
            ty::AutoUnsizeUniq(ty::UnsizeVtable(..)) |
            ty::AutoUnsize(ty::UnsizeVtable(..)) => {
                assert!(n == 1, format!("Expected exactly 1 deref with Uniq \
                                         AutoRefs, found: {}", n));
                let cmt_unadjusted =
                    return_if_err!(self.mc.cat_expr_unadjusted(expr));
                self.delegate_consume(expr.id, expr.span, cmt_unadjusted);
                return;
            }
            _ => {}
        }

        let cmt_derefd = return_if_err!(
            self.mc.cat_expr_autoderefd(expr, n));
        debug!("walk_adjustment: cmt_derefd={}",
               cmt_derefd.repr(self.tcx()));

        match *autoref {
            ty::AutoPtr(r, m, _) => {
                self.delegate.borrow(expr.id,
                                     expr.span,
                                     cmt_derefd,
                                     r,
                                     ty::BorrowKind::from_mutbl(m),
                                     AutoRef);
            }
            ty::AutoUnsizeUniq(_) | ty::AutoUnsize(_) | ty::AutoUnsafe(..) => {}
        }
    }

    fn walk_overloaded_operator(&mut self,
                                expr: &ast::Expr,
                                receiver: &ast::Expr,
                                rhs: Vec<&ast::Expr>,
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

        let r = ty::ReScope(region::CodeExtent::from_node_id(expr.id));
        let bk = ty::ImmBorrow;

        for &arg in &rhs {
            self.borrow_expr(arg, r, bk, OverloadedOperator);
        }
        return true;
    }

    fn arm_move_mode(&mut self, discr_cmt: mc::cmt<'tcx>, arm: &ast::Arm) -> TrackMatchMode {
        let mut mode = Unknown;
        for pat in &arm.pats {
            self.determine_pat_move_mode(discr_cmt.clone(), &**pat, &mut mode);
        }
        mode
    }

    fn walk_arm(&mut self, discr_cmt: mc::cmt<'tcx>, arm: &ast::Arm, mode: MatchMode) {
        for pat in &arm.pats {
            self.walk_pat(discr_cmt.clone(), &**pat, mode);
        }

        if let Some(ref guard) = arm.guard {
            self.consume_expr(&**guard);
        }

        self.consume_expr(&*arm.body);
    }

    /// Walks an pat that occurs in isolation (i.e. top-level of fn
    /// arg or let binding.  *Not* a match arm or nested pat.)
    fn walk_irrefutable_pat(&mut self, cmt_discr: mc::cmt<'tcx>, pat: &ast::Pat) {
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
                               pat: &ast::Pat,
                               mode: &mut TrackMatchMode) {
        debug!("determine_pat_move_mode cmt_discr={} pat={}", cmt_discr.repr(self.tcx()),
               pat.repr(self.tcx()));
        return_if_err!(self.mc.cat_pattern(cmt_discr, pat, |_mc, cmt_pat, pat| {
            let tcx = self.tcx();
            let def_map = &self.tcx().def_map;
            if pat_util::pat_is_binding(def_map, pat) {
                match pat.node {
                    ast::PatIdent(ast::BindByRef(_), _, _) =>
                        mode.lub(BorrowingMatch),
                    ast::PatIdent(ast::BindByValue(_), _, _) => {
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
                pat: &ast::Pat,
                match_mode: MatchMode) {
        debug!("walk_pat cmt_discr={} pat={}", cmt_discr.repr(self.tcx()),
               pat.repr(self.tcx()));

        let mc = &self.mc;
        let typer = self.typer;
        let def_map = &self.tcx().def_map;
        let delegate = &mut self.delegate;
        return_if_err!(mc.cat_pattern(cmt_discr.clone(), pat, |mc, cmt_pat, pat| {
            if pat_util::pat_is_binding(def_map, pat) {
                let tcx = typer.tcx();

                debug!("binding cmt_pat={} pat={} match_mode={:?}",
                       cmt_pat.repr(tcx),
                       pat.repr(tcx),
                       match_mode);

                // pat_ty: the type of the binding being produced.
                let pat_ty = return_if_err!(typer.node_ty(pat.id));

                // Each match binding is effectively an assignment to the
                // binding being produced.
                let def = def_map.borrow()[pat.id].clone();
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
                    ast::PatVec(_, Some(ref slice_pat), _) => {
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

        // Do a second pass over the pattern, calling `matched_pat` on
        // the interior nodes (enum variants and structs), as opposed
        // to the above loop's visit of than the bindings that form
        // the leaves of the pattern tree structure.
        return_if_err!(mc.cat_pattern(cmt_discr, pat, |mc, cmt_pat, pat| {
            let def_map = def_map.borrow();
            let tcx = typer.tcx();

            match pat.node {
                ast::PatEnum(_, _) | ast::PatIdent(_, _, None) | ast::PatStruct(..) => {
                    match def_map.get(&pat.id) {
                        None => {
                            // no definition found: pat is not a
                            // struct or enum pattern.
                        }

                        Some(&def::DefVariant(enum_did, variant_did, _is_struct)) => {
                            let downcast_cmt =
                                if ty::enum_is_univariant(tcx, enum_did) {
                                    cmt_pat
                                } else {
                                    let cmt_pat_ty = cmt_pat.ty;
                                    mc.cat_downcast(pat, cmt_pat, cmt_pat_ty, variant_did)
                                };

                            debug!("variant downcast_cmt={} pat={}",
                                   downcast_cmt.repr(tcx),
                                   pat.repr(tcx));

                            delegate.matched_pat(pat, downcast_cmt, match_mode);
                        }

                        Some(&def::DefStruct(..)) | Some(&def::DefTy(_, false)) => {
                            // A struct (in either the value or type
                            // namespace; we encounter the former on
                            // e.g. patterns for unit structs).

                            debug!("struct cmt_pat={} pat={}",
                                   cmt_pat.repr(tcx),
                                   pat.repr(tcx));

                            delegate.matched_pat(pat, cmt_pat, match_mode);
                        }

                        Some(&def::DefConst(..)) |
                        Some(&def::DefLocal(..)) => {
                            // This is a leaf (i.e. identifier binding
                            // or constant value to match); thus no
                            // `matched_pat` call.
                        }

                        Some(def @ &def::DefTy(_, true)) => {
                            // An enum's type -- should never be in a
                            // pattern.

                            if !tcx.sess.has_errors() {
                                let msg = format!("Pattern has unexpected type: {:?} and type {}",
                                                  def,
                                                  cmt_pat.ty.repr(tcx));
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
                                let msg = format!("Pattern has unexpected def: {:?} and type {}",
                                                  def,
                                                  cmt_pat.ty.repr(tcx));
                                tcx.sess.span_bug(pat.span, &msg[..])
                            }
                        }
                    }
                }

                ast::PatIdent(_, _, Some(_)) => {
                    // Do nothing; this is a binding (not a enum
                    // variant or struct), and the cat_pattern call
                    // will visit the substructure recursively.
                }

                ast::PatWild(_) | ast::PatTup(..) | ast::PatBox(..) |
                ast::PatRegion(..) | ast::PatLit(..) | ast::PatRange(..) |
                ast::PatVec(..) | ast::PatMac(..) => {
                    // Similarly, each of these cases does not
                    // correspond to a enum variant or struct, so we
                    // do not do any `matched_pat` calls for these
                    // cases either.
                }
            }
        }));
    }

    fn walk_captures(&mut self, closure_expr: &ast::Expr) {
        debug!("walk_captures({})", closure_expr.repr(self.tcx()));

        ty::with_freevars(self.tcx(), closure_expr.id, |freevars| {
            for freevar in freevars {
                let id_var = freevar.def.def_id().node;
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
        let var_id = upvar_def.def_id().node;
        let var_ty = try!(self.typer.node_ty(var_id));
        self.mc.cat_def(closure_id, closure_span, var_ty, upvar_def)
    }
}

fn copy_or_move<'tcx>(typer: &mc::Typer<'tcx>,
                      cmt: &mc::cmt<'tcx>,
                      move_reason: MoveReason)
                      -> ConsumeMode
{
    if typer.type_moves_by_default(cmt.span, cmt.ty) {
        Move(move_reason)
    } else {
        Copy
    }
}
