// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Verifies that the types and values of const and static items
// are safe. The rules enforced by this module are:
//
// - For each *mutable* static item, it checks that its **type**:
//     - doesn't have a destructor
//     - doesn't own an owned pointer
//
// - For each *immutable* static item, it checks that its **value**:
//       - doesn't own owned, managed pointers
//       - doesn't contain a struct literal or a call to an enum variant / struct constructor where
//           - the type of the struct/enum has a dtor
//
// Rules Enforced Elsewhere:
// - It's not possible to take the address of a static item with unsafe interior. This is enforced
// by borrowck::gather_loans

use middle::const_eval;
use middle::def;
use middle::expr_use_visitor as euv;
use middle::infer;
use middle::mem_categorization as mc;
use middle::traits;
use middle::ty::{self, Ty};
use util::nodemap::NodeMap;
use util::ppaux;

use syntax::ast;
use syntax::codemap::Span;
use syntax::print::pprust;
use syntax::visit::{self, Visitor};

use std::collections::hash_map::Entry;

// Const qualification, from partial to completely promotable.
bitflags! {
    #[derive(RustcEncodable, RustcDecodable)]
    flags ConstQualif: u8 {
        // Const rvalue which can be placed behind a reference.
        const PURE_CONST          = 0b000000,
        // Inner mutability (can not be placed behind a reference) or behind
        // &mut in a non-global expression. Can be copied from static memory.
        const MUTABLE_MEM         = 0b000001,
        // Constant value with a type that implements Drop. Can be copied
        // from static memory, similar to MUTABLE_MEM.
        const NEEDS_DROP          = 0b000010,
        // Even if the value can be placed in static memory, copying it from
        // there is more expensive than in-place instantiation, and/or it may
        // be too large. This applies to [T; N] and everything containing it.
        // N.B.: references need to clear this flag to not end up on the stack.
        const PREFER_IN_PLACE     = 0b000100,
        // May use more than 0 bytes of memory, doesn't impact the constness
        // directly, but is not allowed to be borrowed mutably in a constant.
        const NON_ZERO_SIZED      = 0b001000,
        // Actually borrowed, has to always be in static memory. Does not
        // propagate, and requires the expression to behave like a 'static
        // lvalue. The set of expressions with this flag is the minimum
        // that have to be promoted.
        const HAS_STATIC_BORROWS  = 0b010000,
        // Invalid const for miscellaneous reasons (e.g. not implemented).
        const NOT_CONST           = 0b100000,

        // Borrowing the expression won't produce &'static T if any of these
        // bits are set, though the value could be copied from static memory
        // if `NOT_CONST` isn't set.
        const NON_STATIC_BORROWS = MUTABLE_MEM.bits | NEEDS_DROP.bits | NOT_CONST.bits
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum Mode {
    Const,
    Static,
    StaticMut,

    // An expression that occurs outside of any constant context
    // (i.e. `const`, `static`, array lengths, etc.). The value
    // can be variable at runtime, but will be promotable to
    // static memory if we can prove it is actually constant.
    Var,
}

struct CheckCrateVisitor<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    mode: Mode,
    qualif: ConstQualif,
    rvalue_borrows: NodeMap<ast::Mutability>
}

impl<'a, 'tcx> CheckCrateVisitor<'a, 'tcx> {
    fn with_mode<F, R>(&mut self, mode: Mode, f: F) -> R where
        F: FnOnce(&mut CheckCrateVisitor<'a, 'tcx>) -> R,
    {
        let (old_mode, old_qualif) = (self.mode, self.qualif);
        self.mode = mode;
        self.qualif = PURE_CONST;
        let r = f(self);
        self.mode = old_mode;
        self.qualif = old_qualif;
        r
    }

    fn with_euv<'b, F, R>(&'b mut self, item_id: Option<ast::NodeId>, f: F) -> R where
        F: for<'t> FnOnce(&mut euv::ExprUseVisitor<'b, 't, 'tcx,
                                    ty::ParameterEnvironment<'a, 'tcx>>) -> R,
    {
        let param_env = match item_id {
            Some(item_id) => ty::ParameterEnvironment::for_item(self.tcx, item_id),
            None => ty::empty_parameter_environment(self.tcx)
        };
        f(&mut euv::ExprUseVisitor::new(self, &param_env))
    }

    fn global_expr(&mut self, mode: Mode, expr: &ast::Expr) -> ConstQualif {
        assert!(mode != Mode::Var);
        match self.tcx.const_qualif_map.borrow_mut().entry(expr.id) {
            Entry::Occupied(entry) => return *entry.get(),
            Entry::Vacant(entry) => {
                // Prevent infinite recursion on re-entry.
                entry.insert(PURE_CONST);
            }
        }
        self.with_mode(mode, |this| {
            this.with_euv(None, |euv| euv.consume_expr(expr));
            this.visit_expr(expr);
            this.qualif
        })
    }

    fn add_qualif(&mut self, qualif: ConstQualif) {
        self.qualif = self.qualif | qualif;
    }

    fn record_borrow(&mut self, id: ast::NodeId, mutbl: ast::Mutability) {
        match self.rvalue_borrows.entry(id) {
            Entry::Occupied(mut entry) => {
                // Merge the two borrows, taking the most demanding
                // one, mutability-wise.
                if mutbl == ast::MutMutable {
                    entry.insert(mutbl);
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(mutbl);
            }
        }
    }

    fn msg(&self) -> &'static str {
        match self.mode {
            Mode::Const => "constant",
            Mode::StaticMut | Mode::Static => "static",
            Mode::Var => unreachable!(),
        }
    }

    fn check_static_mut_type(&self, e: &ast::Expr) {
        let node_ty = ty::node_id_to_type(self.tcx, e.id);
        let tcontents = ty::type_contents(self.tcx, node_ty);

        let suffix = if tcontents.has_dtor() {
            "destructors"
        } else if tcontents.owns_owned() {
            "owned pointers"
        } else {
            return
        };

        self.tcx.sess.span_err(e.span, &format!("mutable statics are not allowed \
                                                 to have {}", suffix));
    }

    fn check_static_type(&self, e: &ast::Expr) {
        let ty = ty::node_id_to_type(self.tcx, e.id);
        let infcx = infer::new_infer_ctxt(self.tcx);
        let mut fulfill_cx = traits::FulfillmentContext::new();
        let cause = traits::ObligationCause::new(e.span, e.id, traits::SharedStatic);
        fulfill_cx.register_builtin_bound(&infcx, ty, ty::BoundSync, cause);
        let env = ty::empty_parameter_environment(self.tcx);
        match fulfill_cx.select_all_or_error(&infcx, &env) {
            Ok(()) => { },
            Err(ref errors) => {
                traits::report_fulfillment_errors(&infcx, errors);
            }
        }
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for CheckCrateVisitor<'a, 'tcx> {
    fn visit_item(&mut self, i: &ast::Item) {
        debug!("visit_item(item={})", pprust::item_to_string(i));
        match i.node {
            ast::ItemStatic(_, ast::MutImmutable, ref expr) => {
                self.check_static_type(&**expr);
                self.global_expr(Mode::Static, &**expr);
            }
            ast::ItemStatic(_, ast::MutMutable, ref expr) => {
                self.check_static_mut_type(&**expr);
                self.global_expr(Mode::StaticMut, &**expr);
            }
            ast::ItemConst(_, ref expr) => {
                self.global_expr(Mode::Const, &**expr);
            }
            ast::ItemEnum(ref enum_definition, _) => {
                for var in &enum_definition.variants {
                    if let Some(ref ex) = var.node.disr_expr {
                        self.global_expr(Mode::Const, &**ex);
                    }
                }
            }
            _ => {
                self.with_mode(Mode::Var, |v| visit::walk_item(v, i));
            }
        }
    }

    fn visit_fn(&mut self,
                fk: visit::FnKind<'v>,
                fd: &'v ast::FnDecl,
                b: &'v ast::Block,
                s: Span,
                fn_id: ast::NodeId) {
        assert!(self.mode == Mode::Var);
        self.with_euv(Some(fn_id), |euv| euv.walk_fn(fd, b));
        visit::walk_fn(self, fk, fd, b, s);
    }

    fn visit_pat(&mut self, p: &ast::Pat) {
        match p.node {
            ast::PatLit(ref lit) => {
                self.global_expr(Mode::Const, &**lit);
            }
            ast::PatRange(ref start, ref end) => {
                self.global_expr(Mode::Const, &**start);
                self.global_expr(Mode::Const, &**end);
            }
            _ => visit::walk_pat(self, p)
        }
    }

    fn visit_expr(&mut self, ex: &ast::Expr) {
        let mut outer = self.qualif;
        self.qualif = PURE_CONST;

        let node_ty = ty::node_id_to_type(self.tcx, ex.id);
        check_expr(self, ex, node_ty);

        // Special-case some expressions to avoid certain flags bubbling up.
        match ex.node {
            ast::ExprCall(ref callee, ref args) => {
                for arg in args.iter() {
                    self.visit_expr(&**arg)
                }

                let inner = self.qualif;
                self.visit_expr(&**callee);
                // The callee's size doesn't count in the call.
                let added = self.qualif - inner;
                self.qualif = inner | (added - NON_ZERO_SIZED);
            }
            ast::ExprRepeat(ref element, _) => {
                self.visit_expr(&**element);
                // The count is checked elsewhere (typeck).
                let count = match node_ty.sty {
                    ty::ty_vec(_, Some(n)) => n,
                    _ => unreachable!()
                };
                // [element; 0] is always zero-sized.
                if count == 0 {
                    self.qualif = self.qualif - (NON_ZERO_SIZED | PREFER_IN_PLACE);
                }
            }
            ast::ExprMatch(ref discr, ref arms, _) => {
                // Compute the most demanding borrow from all the arms'
                // patterns and set that on the discriminator.
                let mut borrow = None;
                for pat in arms.iter().flat_map(|arm| arm.pats.iter()) {
                    let pat_borrow = self.rvalue_borrows.remove(&pat.id);
                    match (borrow, pat_borrow) {
                        (None, _) | (_, Some(ast::MutMutable)) => {
                            borrow = pat_borrow;
                        }
                        _ => {}
                    }
                }
                if let Some(mutbl) = borrow {
                    self.record_borrow(discr.id, mutbl);
                }
                visit::walk_expr(self, ex);
            }
            // Division by zero and overflow checking.
            ast::ExprBinary(op, _, _) => {
                visit::walk_expr(self, ex);
                let div_or_rem = op.node == ast::BiDiv || op.node == ast::BiRem;
                match node_ty.sty {
                    ty::ty_uint(_) | ty::ty_int(_) if div_or_rem => {
                        if !self.qualif.intersects(NOT_CONST) {
                            match const_eval::eval_const_expr_partial(self.tcx, ex, None) {
                                Ok(_) => {}
                                Err(msg) => {
                                    span_err!(self.tcx.sess, msg.span, E0020,
                                              "{} in a constant expression",
                                              msg.description())
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            _ => visit::walk_expr(self, ex)
        }

        // Handle borrows on (or inside the autorefs of) this expression.
        match self.rvalue_borrows.remove(&ex.id) {
            Some(ast::MutImmutable) => {
                // Constants cannot be borrowed if they contain interior mutability as
                // it means that our "silent insertion of statics" could change
                // initializer values (very bad).
                // If the type doesn't have interior mutability, then `MUTABLE_MEM` has
                // propagated from another error, so erroring again would be just noise.
                let tc = ty::type_contents(self.tcx, node_ty);
                if self.qualif.intersects(MUTABLE_MEM) && tc.interior_unsafe() {
                    outer = outer | NOT_CONST;
                    if self.mode != Mode::Var {
                        self.tcx.sess.span_err(ex.span,
                            "cannot borrow a constant which contains \
                             interior mutability, create a static instead");
                    }
                }
                // If the reference has to be 'static, avoid in-place initialization
                // as that will end up pointing to the stack instead.
                if !self.qualif.intersects(NON_STATIC_BORROWS) {
                    self.qualif = self.qualif - PREFER_IN_PLACE;
                    self.add_qualif(HAS_STATIC_BORROWS);
                }
            }
            Some(ast::MutMutable) => {
                // `&mut expr` means expr could be mutated, unless it's zero-sized.
                if self.qualif.intersects(NON_ZERO_SIZED) {
                    if self.mode == Mode::Var {
                        outer = outer | NOT_CONST;
                        self.add_qualif(MUTABLE_MEM);
                    } else {
                        span_err!(self.tcx.sess, ex.span, E0017,
                            "references in {}s may only refer \
                             to immutable values", self.msg())
                    }
                }
                if !self.qualif.intersects(NON_STATIC_BORROWS) {
                    self.add_qualif(HAS_STATIC_BORROWS);
                }
            }
            None => {}
        }
        self.tcx.const_qualif_map.borrow_mut().insert(ex.id, self.qualif);
        // Don't propagate certain flags.
        self.qualif = outer | (self.qualif - HAS_STATIC_BORROWS);
    }
}

/// This function is used to enforce the constraints on
/// const/static items. It walks through the *value*
/// of the item walking down the expression and evaluating
/// every nested expression. If the expression is not part
/// of a const/static item, it is qualified for promotion
/// instead of producing errors.
fn check_expr<'a, 'tcx>(v: &mut CheckCrateVisitor<'a, 'tcx>,
                        e: &ast::Expr, node_ty: Ty<'tcx>) {
    match node_ty.sty {
        ty::ty_struct(did, _) |
        ty::ty_enum(did, _) if ty::has_dtor(v.tcx, did) => {
            v.add_qualif(NEEDS_DROP);
            if v.mode != Mode::Var {
                v.tcx.sess.span_err(e.span,
                                    &format!("{}s are not allowed to have destructors",
                                             v.msg()));
            }
        }
        _ => {}
    }

    let method_call = ty::MethodCall::expr(e.id);
    match e.node {
        ast::ExprUnary(..) |
        ast::ExprBinary(..) |
        ast::ExprIndex(..) if v.tcx.method_map.borrow().contains_key(&method_call) => {
            v.add_qualif(NOT_CONST);
            if v.mode != Mode::Var {
                span_err!(v.tcx.sess, e.span, E0011,
                            "user-defined operators are not allowed in {}s", v.msg());
            }
        }
        ast::ExprBox(..) |
        ast::ExprUnary(ast::UnUniq, _) => {
            v.add_qualif(NOT_CONST);
            if v.mode != Mode::Var {
                span_err!(v.tcx.sess, e.span, E0010,
                          "allocations are not allowed in {}s", v.msg());
            }
        }
        ast::ExprUnary(ast::UnDeref, ref ptr) => {
            match ty::node_id_to_type(v.tcx, ptr.id).sty {
                ty::ty_ptr(_) => {
                    // This shouldn't be allowed in constants at all.
                    v.add_qualif(NOT_CONST);
                }
                _ => {}
            }
        }
        ast::ExprCast(ref from, _) => {
            let toty = ty::expr_ty(v.tcx, e);
            let fromty = ty::expr_ty(v.tcx, &**from);
            let is_legal_cast =
                ty::type_is_numeric(toty) ||
                ty::type_is_unsafe_ptr(toty) ||
                (ty::type_is_bare_fn(toty) && ty::type_is_bare_fn_item(fromty));
            if !is_legal_cast {
                v.add_qualif(NOT_CONST);
                if v.mode != Mode::Var {
                    span_err!(v.tcx.sess, e.span, E0012,
                              "can not cast to `{}` in {}s",
                              ppaux::ty_to_string(v.tcx, toty), v.msg());
                }
            }
            if ty::type_is_unsafe_ptr(fromty) && ty::type_is_numeric(toty) {
                v.add_qualif(NOT_CONST);
                if v.mode != Mode::Var {
                    span_err!(v.tcx.sess, e.span, E0018,
                              "can not cast a pointer to an integer in {}s", v.msg());
                }
            }
        }
        ast::ExprPath(..) => {
            let def = v.tcx.def_map.borrow().get(&e.id).map(|d| d.full_def());
            match def {
                Some(def::DefVariant(_, _, _)) => {
                    // Count the discriminator or function pointer.
                    v.add_qualif(NON_ZERO_SIZED);
                }
                Some(def::DefStruct(_)) => {
                    if let ty::ty_bare_fn(..) = node_ty.sty {
                        // Count the function pointer.
                        v.add_qualif(NON_ZERO_SIZED);
                    }
                }
                Some(def::DefFn(..)) | Some(def::DefMethod(..)) => {
                    // Count the function pointer.
                    v.add_qualif(NON_ZERO_SIZED);
                }
                Some(def::DefStatic(..)) => {
                    match v.mode {
                        Mode::Static | Mode::StaticMut => {}
                        Mode::Const => {
                            span_err!(v.tcx.sess, e.span, E0013,
                                "constants cannot refer to other statics, \
                                 insert an intermediate constant instead");
                        }
                        Mode::Var => v.add_qualif(NOT_CONST)
                    }
                }
                Some(def::DefConst(did)) => {
                    if let Some(expr) = const_eval::lookup_const_by_id(v.tcx, did) {
                        let inner = v.global_expr(Mode::Const, expr);
                        v.add_qualif(inner);
                    } else {
                        v.tcx.sess.span_bug(e.span, "DefConst doesn't point \
                                                     to an ItemConst");
                    }
                }
                def => {
                    v.add_qualif(NOT_CONST);
                    if v.mode != Mode::Var {
                        debug!("(checking const) found bad def: {:?}", def);
                        span_err!(v.tcx.sess, e.span, E0014,
                                  "paths in {}s may only refer to constants \
                                   or functions", v.msg());
                    }
                }
            }
        }
        ast::ExprCall(ref callee, _) => {
            let mut callee = &**callee;
            loop {
                callee = match callee.node {
                    ast::ExprParen(ref inner) => &**inner,
                    ast::ExprBlock(ref block) => match block.expr {
                        Some(ref tail) => &**tail,
                        None => break
                    },
                    _ => break
                };
            }
            let def = v.tcx.def_map.borrow().get(&callee.id).map(|d| d.full_def());
            match def {
                Some(def::DefStruct(..)) => {}
                Some(def::DefVariant(..)) => {
                    // Count the discriminator.
                    v.add_qualif(NON_ZERO_SIZED);
                }
                _ => {
                    v.add_qualif(NOT_CONST);
                    if v.mode != Mode::Var {
                        span_err!(v.tcx.sess, e.span, E0015,
                                  "function calls in {}s are limited to \
                                   struct and enum constructors", v.msg());
                    }
                }
            }
        }
        ast::ExprBlock(ref block) => {
            // Check all statements in the block
            let mut block_span_err = |span| {
                v.add_qualif(NOT_CONST);
                if v.mode != Mode::Var {
                    span_err!(v.tcx.sess, span, E0016,
                              "blocks in {}s are limited to items and \
                               tail expressions", v.msg());
                }
            };
            for stmt in &block.stmts {
                match stmt.node {
                    ast::StmtDecl(ref decl, _) => {
                        match decl.node {
                            ast::DeclLocal(_) => block_span_err(decl.span),

                            // Item statements are allowed
                            ast::DeclItem(_) => {}
                        }
                    }
                    ast::StmtExpr(ref expr, _) => block_span_err(expr.span),
                    ast::StmtSemi(ref semi, _) => block_span_err(semi.span),
                    ast::StmtMac(..) => {
                        v.tcx.sess.span_bug(e.span, "unexpanded statement \
                                                     macro in const?!")
                    }
                }
            }
        }
        ast::ExprStruct(..) => {
            let did = v.tcx.def_map.borrow().get(&e.id).map(|def| def.def_id());
            if did == v.tcx.lang_items.unsafe_cell_type() {
                v.add_qualif(MUTABLE_MEM);
            }
        }

        ast::ExprLit(_) |
        ast::ExprAddrOf(..) => {
            v.add_qualif(NON_ZERO_SIZED);
        }

        ast::ExprRepeat(..) => {
            v.add_qualif(PREFER_IN_PLACE);
        }

        ast::ExprClosure(..) => {
            // Paths in constant constexts cannot refer to local variables,
            // as there are none, and thus closures can't have upvars there.
            if ty::with_freevars(v.tcx, e.id, |fv| !fv.is_empty()) {
                assert!(v.mode == Mode::Var,
                        "global closures can't capture anything");
                v.add_qualif(NOT_CONST);
            }
        }

        ast::ExprUnary(..) |
        ast::ExprBinary(..) |
        ast::ExprIndex(..) |
        ast::ExprField(..) |
        ast::ExprTupField(..) |
        ast::ExprVec(_) |
        ast::ExprParen(..) |
        ast::ExprTup(..) => {}

        // Conditional control flow (possible to implement).
        ast::ExprMatch(..) |
        ast::ExprIf(..) |
        ast::ExprIfLet(..) |

        // Loops (not very meaningful in constants).
        ast::ExprWhile(..) |
        ast::ExprWhileLet(..) |
        ast::ExprForLoop(..) |
        ast::ExprLoop(..) |

        // More control flow (also not very meaningful).
        ast::ExprBreak(_) |
        ast::ExprAgain(_) |
        ast::ExprRet(_) |

        // Miscellaneous expressions that could be implemented.
        ast::ExprRange(..) |

        // Various other expressions.
        ast::ExprMethodCall(..) |
        ast::ExprAssign(..) |
        ast::ExprAssignOp(..) |
        ast::ExprInlineAsm(_) |
        ast::ExprMac(_) => {
            v.add_qualif(NOT_CONST);
            if v.mode != Mode::Var {
                span_err!(v.tcx.sess, e.span, E0019,
                          "{} contains unimplemented expression type", v.msg());
            }
        }
    }
}

pub fn check_crate(tcx: &ty::ctxt) {
    visit::walk_crate(&mut CheckCrateVisitor {
        tcx: tcx,
        mode: Mode::Var,
        qualif: NOT_CONST,
        rvalue_borrows: NodeMap()
    }, tcx.map.krate());

    tcx.sess.abort_if_errors();
}

impl<'a, 'tcx> euv::Delegate<'tcx> for CheckCrateVisitor<'a, 'tcx> {
    fn consume(&mut self,
               _consume_id: ast::NodeId,
               consume_span: Span,
               cmt: mc::cmt,
               _mode: euv::ConsumeMode) {
        let mut cur = &cmt;
        loop {
            match cur.cat {
                mc::cat_static_item => {
                    if self.mode != Mode::Var {
                        // statics cannot be consumed by value at any time, that would imply
                        // that they're an initializer (what a const is for) or kept in sync
                        // over time (not feasible), so deny it outright.
                        self.tcx.sess.span_err(consume_span,
                            "cannot refer to other statics by value, use the \
                             address-of operator or a constant instead");
                    }
                    break;
                }
                mc::cat_deref(ref cmt, _, _) |
                mc::cat_downcast(ref cmt, _) |
                mc::cat_interior(ref cmt, _) => cur = cmt,

                mc::cat_rvalue(..) |
                mc::cat_upvar(..) |
                mc::cat_local(..) => break
            }
        }
    }
    fn borrow(&mut self,
              borrow_id: ast::NodeId,
              borrow_span: Span,
              cmt: mc::cmt<'tcx>,
              _loan_region: ty::Region,
              bk: ty::BorrowKind,
              loan_cause: euv::LoanCause) {
        let mut cur = &cmt;
        let mut is_interior = false;
        loop {
            match cur.cat {
                mc::cat_rvalue(..) => {
                    if loan_cause == euv::MatchDiscriminant {
                        // Ignore the dummy immutable borrow created by EUV.
                        break;
                    }
                    let mutbl = bk.to_mutbl_lossy();
                    if mutbl == ast::MutMutable && self.mode == Mode::StaticMut {
                        // Mutable slices are the only `&mut` allowed in globals,
                        // but only in `static mut`, nowhere else.
                        match cmt.ty.sty {
                            ty::ty_vec(_, _) => break,
                            _ => {}
                        }
                    }
                    self.record_borrow(borrow_id, mutbl);
                    break;
                }
                mc::cat_static_item => {
                    if is_interior && self.mode != Mode::Var {
                        // Borrowed statics can specifically *only* have their address taken,
                        // not any number of other borrows such as borrowing fields, reading
                        // elements of an array, etc.
                        self.tcx.sess.span_err(borrow_span,
                            "cannot refer to the interior of another \
                             static, use a constant instead");
                    }
                    break;
                }
                mc::cat_deref(ref cmt, _, _) |
                mc::cat_downcast(ref cmt, _) |
                mc::cat_interior(ref cmt, _) => {
                    is_interior = true;
                    cur = cmt;
                }

                mc::cat_upvar(..) |
                mc::cat_local(..) => break
            }
        }
    }

    fn decl_without_init(&mut self,
                         _id: ast::NodeId,
                         _span: Span) {}
    fn mutate(&mut self,
              _assignment_id: ast::NodeId,
              _assignment_span: Span,
              _assignee_cmt: mc::cmt,
              _mode: euv::MutateMode) {}

    fn matched_pat(&mut self,
                   _: &ast::Pat,
                   _: mc::cmt,
                   _: euv::MatchMode) {}

    fn consume_pat(&mut self,
                   _consume_pat: &ast::Pat,
                   _cmt: mc::cmt,
                   _mode: euv::ConsumeMode) {}
}
