// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * # Categorization
 *
 * The job of the categorization module is to analyze an expression to
 * determine what kind of memory is used in evaluating it (for example,
 * where dereferences occur and what kind of pointer is dereferenced;
 * whether the memory is mutable; etc)
 *
 * Categorization effectively transforms all of our expressions into
 * expressions of the following forms (the actual enum has many more
 * possibilities, naturally, but they are all variants of these base
 * forms):
 *
 *     E = rvalue    // some computed rvalue
 *       | x         // address of a local variable or argument
 *       | *E        // deref of a ptr
 *       | E.comp    // access to an interior component
 *
 * Imagine a routine ToAddr(Expr) that evaluates an expression and returns an
 * address where the result is to be found.  If Expr is an lvalue, then this
 * is the address of the lvalue.  If Expr is an rvalue, this is the address of
 * some temporary spot in memory where the result is stored.
 *
 * Now, cat_expr() classifies the expression Expr and the address A=ToAddr(Expr)
 * as follows:
 *
 * - cat: what kind of expression was this?  This is a subset of the
 *   full expression forms which only includes those that we care about
 *   for the purpose of the analysis.
 * - mutbl: mutability of the address A
 * - ty: the type of data found at the address A
 *
 * The resulting categorization tree differs somewhat from the expressions
 * themselves.  For example, auto-derefs are explicit.  Also, an index a[b] is
 * decomposed into two operations: a dereference to reach the array data and
 * then an index to jump forward to the relevant item.
 *
 * ## By-reference upvars
 *
 * One part of the translation which may be non-obvious is that we translate
 * closure upvars into the dereference of a borrowed pointer; this more closely
 * resembles the runtime translation. So, for example, if we had:
 *
 *     let mut x = 3;
 *     let y = 5;
 *     let inc = || x += y;
 *
 * Then when we categorize `x` (*within* the closure) we would yield a
 * result of `*x'`, effectively, where `x'` is a `cat_upvar` reference
 * tied to `x`. The type of `x'` will be a borrowed pointer.
 */

#![allow(non_camel_case_types)]

use middle::def;
use middle::ty;
use middle::typeck;
use util::nodemap::{DefIdMap, NodeMap};
use util::ppaux::{ty_to_string, Repr};

use syntax::ast::{MutImmutable, MutMutable};
use syntax::ast;
use syntax::ast_map;
use syntax::codemap::Span;
use syntax::print::pprust;
use syntax::parse::token;

use std::cell::RefCell;
use std::rc::Rc;

#[deriving(Clone, PartialEq, Show)]
pub enum categorization {
    cat_rvalue(ty::Region),            // temporary val, argument is its scope
    cat_static_item,
    cat_upvar(Upvar),                  // upvar referenced by closure env
    cat_local(ast::NodeId),            // local variable
    cat_deref(cmt, uint, PointerKind), // deref of a ptr
    cat_interior(cmt, InteriorKind),   // something interior: field, tuple, etc
    cat_downcast(cmt),                 // selects a particular enum variant (*1)
    cat_discr(cmt, ast::NodeId),       // match discriminant (see preserve())

    // (*1) downcast is only required if the enum has more than one variant
}

// Represents any kind of upvar
#[deriving(Clone, PartialEq, Show)]
pub struct Upvar {
    pub id: ty::UpvarId,
    // Unboxed closure kinds are used even for old-style closures for simplicity
    pub kind: ty::UnboxedClosureKind,
    // Is this from an unboxed closure?  Used only for diagnostics.
    pub is_unboxed: bool
}

// different kinds of pointers:
#[deriving(Clone, PartialEq, Eq, Hash, Show)]
pub enum PointerKind {
    OwnedPtr,
    BorrowedPtr(ty::BorrowKind, ty::Region),
    Implicit(ty::BorrowKind, ty::Region),     // Implicit deref of a borrowed ptr.
    UnsafePtr(ast::Mutability)
}

// We use the term "interior" to mean "something reachable from the
// base without a pointer dereference", e.g. a field
#[deriving(Clone, PartialEq, Eq, Hash, Show)]
pub enum InteriorKind {
    InteriorField(FieldName),
    InteriorElement(ElementKind),
}

#[deriving(Clone, PartialEq, Eq, Hash, Show)]
pub enum FieldName {
    NamedField(ast::Name),
    PositionalField(uint)
}

#[deriving(Clone, PartialEq, Eq, Hash, Show)]
pub enum ElementKind {
    VecElement,
    OtherElement,
}

#[deriving(Clone, PartialEq, Eq, Hash, Show)]
pub enum MutabilityCategory {
    McImmutable, // Immutable.
    McDeclared,  // Directly declared as mutable.
    McInherited, // Inherited from the fact that owner is mutable.
}

// A note about the provenance of a `cmt`.  This is used for
// special-case handling of upvars such as mutability inference.
// Upvar categorization can generate a variable number of nested
// derefs.  The note allows detecting them without deep pattern
// matching on the categorization.
#[deriving(Clone, PartialEq, Show)]
pub enum Note {
    NoteClosureEnv(ty::UpvarId), // Deref through closure env
    NoteUpvarRef(ty::UpvarId),   // Deref through by-ref upvar
    NoteNone                     // Nothing special
}

// `cmt`: "Category, Mutability, and Type".
//
// a complete categorization of a value indicating where it originated
// and how it is located, as well as the mutability of the memory in
// which the value is stored.
//
// *WARNING* The field `cmt.type` is NOT necessarily the same as the
// result of `node_id_to_type(cmt.id)`. This is because the `id` is
// always the `id` of the node producing the type; in an expression
// like `*x`, the type of this deref node is the deref'd type (`T`),
// but in a pattern like `@x`, the `@x` pattern is again a
// dereference, but its type is the type *before* the dereference
// (`@T`). So use `cmt.ty` to find the type of the value in a consistent
// fashion. For more details, see the method `cat_pattern`
#[deriving(Clone, PartialEq, Show)]
pub struct cmt_ {
    pub id: ast::NodeId,          // id of expr/pat producing this value
    pub span: Span,                // span of same expr/pat
    pub cat: categorization,       // categorization of expr
    pub mutbl: MutabilityCategory, // mutability of expr as lvalue
    pub ty: ty::t,                 // type of the expr (*see WARNING above*)
    pub note: Note,                // Note about the provenance of this cmt
}

pub type cmt = Rc<cmt_>;

// We pun on *T to mean both actual deref of a ptr as well
// as accessing of components:
pub enum deref_kind {
    deref_ptr(PointerKind),
    deref_interior(InteriorKind),
}

// Categorizes a derefable type.  Note that we include vectors and strings as
// derefable (we model an index as the combination of a deref and then a
// pointer adjustment).
pub fn opt_deref_kind(t: ty::t) -> Option<deref_kind> {
    match ty::get(t).sty {
        ty::ty_uniq(_) |
        ty::ty_closure(box ty::ClosureTy {store: ty::UniqTraitStore, ..}) => {
            Some(deref_ptr(OwnedPtr))
        }

        ty::ty_rptr(r, mt) => {
            let kind = ty::BorrowKind::from_mutbl(mt.mutbl);
            Some(deref_ptr(BorrowedPtr(kind, r)))
        }

        ty::ty_closure(box ty::ClosureTy {
                store: ty::RegionTraitStore(r, _),
                ..
            }) => {
            Some(deref_ptr(BorrowedPtr(ty::ImmBorrow, r)))
        }

        ty::ty_ptr(ref mt) => {
            Some(deref_ptr(UnsafePtr(mt.mutbl)))
        }

        ty::ty_enum(..) |
        ty::ty_struct(..) => { // newtype
            Some(deref_interior(InteriorField(PositionalField(0))))
        }

        ty::ty_vec(_, _) | ty::ty_str => {
            Some(deref_interior(InteriorElement(element_kind(t))))
        }

        _ => None
    }
}

pub fn deref_kind(tcx: &ty::ctxt, t: ty::t) -> deref_kind {
    debug!("deref_kind {}", ty_to_string(tcx, t));
    match opt_deref_kind(t) {
      Some(k) => k,
      None => {
        tcx.sess.bug(
            format!("deref_kind() invoked on non-derefable type {}",
                    ty_to_string(tcx, t)).as_slice());
      }
    }
}

pub trait ast_node {
    fn id(&self) -> ast::NodeId;
    fn span(&self) -> Span;
}

impl ast_node for ast::Expr {
    fn id(&self) -> ast::NodeId { self.id }
    fn span(&self) -> Span { self.span }
}

impl ast_node for ast::Pat {
    fn id(&self) -> ast::NodeId { self.id }
    fn span(&self) -> Span { self.span }
}

pub struct MemCategorizationContext<'t,TYPER:'t> {
    typer: &'t TYPER
}

pub type McResult<T> = Result<T, ()>;

/**
 * The `Typer` trait provides the interface for the mem-categorization
 * module to the results of the type check. It can be used to query
 * the type assigned to an expression node, to inquire after adjustments,
 * and so on.
 *
 * This interface is needed because mem-categorization is used from
 * two places: `regionck` and `borrowck`. `regionck` executes before
 * type inference is complete, and hence derives types and so on from
 * intermediate tables.  This also implies that type errors can occur,
 * and hence `node_ty()` and friends return a `Result` type -- any
 * error will propagate back up through the mem-categorization
 * routines.
 *
 * In the borrow checker, in contrast, type checking is complete and we
 * know that no errors have occurred, so we simply consult the tcx and we
 * can be sure that only `Ok` results will occur.
 */
pub trait Typer<'tcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx>;
    fn node_ty(&self, id: ast::NodeId) -> McResult<ty::t>;
    fn node_method_ty(&self, method_call: typeck::MethodCall) -> Option<ty::t>;
    fn adjustments<'a>(&'a self) -> &'a RefCell<NodeMap<ty::AutoAdjustment>>;
    fn is_method_call(&self, id: ast::NodeId) -> bool;
    fn temporary_scope(&self, rvalue_id: ast::NodeId) -> Option<ast::NodeId>;
    fn upvar_borrow(&self, upvar_id: ty::UpvarId) -> ty::UpvarBorrow;
    fn capture_mode(&self, closure_expr_id: ast::NodeId)
                    -> ast::CaptureClause;
    fn unboxed_closures<'a>(&'a self)
                        -> &'a RefCell<DefIdMap<ty::UnboxedClosure>>;
}

impl MutabilityCategory {
    pub fn from_mutbl(m: ast::Mutability) -> MutabilityCategory {
        match m {
            MutImmutable => McImmutable,
            MutMutable => McDeclared
        }
    }

    pub fn from_borrow_kind(borrow_kind: ty::BorrowKind) -> MutabilityCategory {
        match borrow_kind {
            ty::ImmBorrow => McImmutable,
            ty::UniqueImmBorrow => McImmutable,
            ty::MutBorrow => McDeclared,
        }
    }

    pub fn from_pointer_kind(base_mutbl: MutabilityCategory,
                             ptr: PointerKind) -> MutabilityCategory {
        match ptr {
            OwnedPtr => {
                base_mutbl.inherit()
            }
            BorrowedPtr(borrow_kind, _) | Implicit(borrow_kind, _) => {
                MutabilityCategory::from_borrow_kind(borrow_kind)
            }
            UnsafePtr(m) => {
                MutabilityCategory::from_mutbl(m)
            }
        }
    }

    fn from_local(tcx: &ty::ctxt, id: ast::NodeId) -> MutabilityCategory {
        match tcx.map.get(id) {
            ast_map::NodeLocal(p) | ast_map::NodeArg(p) => match p.node {
                ast::PatIdent(bind_mode, _, _) => {
                    if bind_mode == ast::BindByValue(ast::MutMutable) {
                        McDeclared
                    } else {
                        McImmutable
                    }
                }
                _ => tcx.sess.span_bug(p.span, "expected identifier pattern")
            },
            _ => tcx.sess.span_bug(tcx.map.span(id), "expected identifier pattern")
        }
    }

    pub fn inherit(&self) -> MutabilityCategory {
        match *self {
            McImmutable => McImmutable,
            McDeclared => McInherited,
            McInherited => McInherited,
        }
    }

    pub fn is_mutable(&self) -> bool {
        match *self {
            McImmutable => false,
            McInherited => true,
            McDeclared => true,
        }
    }

    pub fn is_immutable(&self) -> bool {
        match *self {
            McImmutable => true,
            McDeclared | McInherited => false
        }
    }

    pub fn to_user_str(&self) -> &'static str {
        match *self {
            McDeclared | McInherited => "mutable",
            McImmutable => "immutable",
        }
    }
}

macro_rules! if_ok(
    ($inp: expr) => (
        match $inp {
            Ok(v) => { v }
            Err(e) => { return Err(e); }
        }
    )
)

impl<'t,'tcx,TYPER:Typer<'tcx>> MemCategorizationContext<'t,TYPER> {
    pub fn new(typer: &'t TYPER) -> MemCategorizationContext<'t,TYPER> {
        MemCategorizationContext { typer: typer }
    }

    fn tcx(&self) -> &'t ty::ctxt<'tcx> {
        self.typer.tcx()
    }

    fn expr_ty(&self, expr: &ast::Expr) -> McResult<ty::t> {
        self.typer.node_ty(expr.id)
    }

    fn expr_ty_adjusted(&self, expr: &ast::Expr) -> McResult<ty::t> {
        let unadjusted_ty = if_ok!(self.expr_ty(expr));
        Ok(ty::adjust_ty(self.tcx(), expr.span, expr.id, unadjusted_ty,
                         self.typer.adjustments().borrow().find(&expr.id),
                         |method_call| self.typer.node_method_ty(method_call)))
    }

    fn node_ty(&self, id: ast::NodeId) -> McResult<ty::t> {
        self.typer.node_ty(id)
    }

    fn pat_ty(&self, pat: &ast::Pat) -> McResult<ty::t> {
        self.typer.node_ty(pat.id)
    }

    pub fn cat_expr(&self, expr: &ast::Expr) -> McResult<cmt> {
        match self.typer.adjustments().borrow().find(&expr.id) {
            None => {
                // No adjustments.
                self.cat_expr_unadjusted(expr)
            }

            Some(adjustment) => {
                match *adjustment {
                    ty::AdjustAddEnv(..) => {
                        debug!("cat_expr(AdjustAddEnv): {}",
                               expr.repr(self.tcx()));
                        // Convert a bare fn to a closure by adding NULL env.
                        // Result is an rvalue.
                        let expr_ty = if_ok!(self.expr_ty_adjusted(expr));
                        Ok(self.cat_rvalue_node(expr.id(), expr.span(), expr_ty))
                    }

                    ty::AdjustDerefRef(
                        ty::AutoDerefRef {
                            autoref: Some(_), ..}) => {
                        debug!("cat_expr(AdjustDerefRef): {}",
                               expr.repr(self.tcx()));
                        // Equivalent to &*expr or something similar.
                        // Result is an rvalue.
                        let expr_ty = if_ok!(self.expr_ty_adjusted(expr));
                        Ok(self.cat_rvalue_node(expr.id(), expr.span(), expr_ty))
                    }

                    ty::AdjustDerefRef(
                        ty::AutoDerefRef {
                            autoref: None, autoderefs: autoderefs}) => {
                        // Equivalent to *expr or something similar.
                        self.cat_expr_autoderefd(expr, autoderefs)
                    }
                }
            }
        }
    }

    pub fn cat_expr_autoderefd(&self,
                               expr: &ast::Expr,
                               autoderefs: uint)
                               -> McResult<cmt> {
        let mut cmt = if_ok!(self.cat_expr_unadjusted(expr));
        debug!("cat_expr_autoderefd: autoderefs={}, cmt={}",
               autoderefs,
               cmt.repr(self.tcx()));
        for deref in range(1u, autoderefs + 1) {
            cmt = self.cat_deref(expr, cmt, deref, false);
        }
        return Ok(cmt);
    }

    pub fn cat_expr_unadjusted(&self, expr: &ast::Expr) -> McResult<cmt> {
        debug!("cat_expr: id={} expr={}", expr.id, expr.repr(self.tcx()));

        let expr_ty = if_ok!(self.expr_ty(expr));
        match expr.node {
          ast::ExprUnary(ast::UnDeref, ref e_base) => {
            let base_cmt = if_ok!(self.cat_expr(&**e_base));
            Ok(self.cat_deref(expr, base_cmt, 0, false))
          }

          ast::ExprField(ref base, f_name, _) => {
            let base_cmt = if_ok!(self.cat_expr(&**base));
            debug!("cat_expr(cat_field): id={} expr={} base={}",
                   expr.id,
                   expr.repr(self.tcx()),
                   base_cmt.repr(self.tcx()));
            Ok(self.cat_field(expr, base_cmt, f_name.node, expr_ty))
          }

          ast::ExprTupField(ref base, idx, _) => {
            let base_cmt = if_ok!(self.cat_expr(&**base));
            Ok(self.cat_tup_field(expr, base_cmt, idx.node, expr_ty))
          }

          ast::ExprIndex(ref base, _) => {
            let method_call = typeck::MethodCall::expr(expr.id());
            match self.typer.node_method_ty(method_call) {
                Some(method_ty) => {
                    // If this is an index implemented by a method call, then it will
                    // include an implicit deref of the result.
                    let ret_ty = ty::ty_fn_ret(method_ty);
                    Ok(self.cat_deref(expr,
                                      self.cat_rvalue_node(expr.id(),
                                                           expr.span(),
                                                           ret_ty), 1, true))
                }
                None => {
                    let base_cmt = if_ok!(self.cat_expr(&**base));
                    Ok(self.cat_index(expr, base_cmt))
                }
            }
          }

          ast::ExprPath(_) => {
            let def = *self.tcx().def_map.borrow().get(&expr.id);
            self.cat_def(expr.id, expr.span, expr_ty, def)
          }

          ast::ExprParen(ref e) => {
            self.cat_expr(&**e)
          }

          ast::ExprAddrOf(..) | ast::ExprCall(..) |
          ast::ExprAssign(..) | ast::ExprAssignOp(..) |
          ast::ExprFnBlock(..) | ast::ExprProc(..) |
          ast::ExprUnboxedFn(..) | ast::ExprRet(..) |
          ast::ExprUnary(..) | ast::ExprSlice(..) |
          ast::ExprMethodCall(..) | ast::ExprCast(..) |
          ast::ExprVec(..) | ast::ExprTup(..) | ast::ExprIf(..) |
          ast::ExprBinary(..) | ast::ExprWhile(..) |
          ast::ExprBlock(..) | ast::ExprLoop(..) | ast::ExprMatch(..) |
          ast::ExprLit(..) | ast::ExprBreak(..) | ast::ExprMac(..) |
          ast::ExprAgain(..) | ast::ExprStruct(..) | ast::ExprRepeat(..) |
          ast::ExprInlineAsm(..) | ast::ExprBox(..) |
          ast::ExprForLoop(..) => {
            Ok(self.cat_rvalue_node(expr.id(), expr.span(), expr_ty))
          }

          ast::ExprIfLet(..) => {
            self.tcx().sess.span_bug(expr.span, "non-desugared ExprIfLet");
          }
          ast::ExprWhileLet(..) => {
            self.tcx().sess.span_bug(expr.span, "non-desugared ExprWhileLet");
          }
        }
    }

    pub fn cat_def(&self,
                   id: ast::NodeId,
                   span: Span,
                   expr_ty: ty::t,
                   def: def::Def)
                   -> McResult<cmt> {
        debug!("cat_def: id={} expr={} def={}",
               id, expr_ty.repr(self.tcx()), def);

        match def {
          def::DefStruct(..) | def::DefVariant(..) | def::DefFn(..) |
          def::DefStaticMethod(..) | def::DefConst(..) => {
                Ok(self.cat_rvalue_node(id, span, expr_ty))
          }
          def::DefMod(_) | def::DefForeignMod(_) | def::DefUse(_) |
          def::DefTrait(_) | def::DefTy(..) | def::DefPrimTy(_) |
          def::DefTyParam(..) | def::DefTyParamBinder(..) | def::DefRegion(_) |
          def::DefLabel(_) | def::DefSelfTy(..) | def::DefMethod(..) |
          def::DefAssociatedTy(..) => {
              Ok(Rc::new(cmt_ {
                  id:id,
                  span:span,
                  cat:cat_static_item,
                  mutbl: McImmutable,
                  ty:expr_ty,
                  note: NoteNone
              }))
          }

          def::DefStatic(_, mutbl) => {
              Ok(Rc::new(cmt_ {
                  id:id,
                  span:span,
                  cat:cat_static_item,
                  mutbl: if mutbl { McDeclared } else { McImmutable},
                  ty:expr_ty,
                  note: NoteNone
              }))
          }

          def::DefUpvar(var_id, fn_node_id, _) => {
              let ty = if_ok!(self.node_ty(fn_node_id));
              match ty::get(ty).sty {
                  ty::ty_closure(ref closure_ty) => {
                      // Translate old closure type info into unboxed
                      // closure kind/capture mode
                      let (mode, kind) = match (closure_ty.store, closure_ty.onceness) {
                          // stack closure
                          (ty::RegionTraitStore(..), ast::Many) => {
                              (ast::CaptureByRef, ty::FnMutUnboxedClosureKind)
                          }
                          // proc or once closure
                          (_, ast::Once) => {
                              (ast::CaptureByValue, ty::FnOnceUnboxedClosureKind)
                          }
                          // There should be no such old closure type
                          (ty::UniqTraitStore, ast::Many) => {
                              self.tcx().sess.span_bug(span, "Impossible closure type");
                          }
                      };
                      self.cat_upvar(id, span, var_id, fn_node_id, kind, mode, false)
                  }
                  ty::ty_unboxed_closure(closure_id, _) => {
                      let unboxed_closures = self.typer.unboxed_closures().borrow();
                      let kind = unboxed_closures.get(&closure_id).kind;
                      let mode = self.typer.capture_mode(fn_node_id);
                      self.cat_upvar(id, span, var_id, fn_node_id, kind, mode, true)
                  }
                  _ => {
                      self.tcx().sess.span_bug(
                          span,
                          format!("Upvar of non-closure {} - {}",
                                  fn_node_id,
                                  ty.repr(self.tcx())).as_slice());
                  }
              }
          }

          def::DefLocal(vid) => {
            Ok(Rc::new(cmt_ {
                id: id,
                span: span,
                cat: cat_local(vid),
                mutbl: MutabilityCategory::from_local(self.tcx(), vid),
                ty: expr_ty,
                note: NoteNone
            }))
          }
        }
    }

    // Categorize an upvar, complete with invisible derefs of closure
    // environment and upvar reference as appropriate.
    fn cat_upvar(&self,
                 id: ast::NodeId,
                 span: Span,
                 var_id: ast::NodeId,
                 fn_node_id: ast::NodeId,
                 kind: ty::UnboxedClosureKind,
                 mode: ast::CaptureClause,
                 is_unboxed: bool)
                 -> McResult<cmt> {
        // An upvar can have up to 3 components.  The base is a
        // `cat_upvar`.  Next, we add a deref through the implicit
        // environment pointer with an anonymous free region 'env and
        // appropriate borrow kind for closure kinds that take self by
        // reference.  Finally, if the upvar was captured
        // by-reference, we add a deref through that reference.  The
        // region of this reference is an inference variable 'up that
        // was previously generated and recorded in the upvar borrow
        // map.  The borrow kind bk is inferred by based on how the
        // upvar is used.
        //
        // This results in the following table for concrete closure
        // types:
        //
        //                | move                 | ref
        // ---------------+----------------------+-------------------------------
        // Fn             | copied -> &'env      | upvar -> &'env -> &'up bk
        // FnMut          | copied -> &'env mut  | upvar -> &'env mut -> &'up bk
        // FnOnce         | copied               | upvar -> &'up bk
        // old stack      | N/A                  | upvar -> &'env mut -> &'up bk
        // old proc/once  | copied               | N/A
        let upvar_id = ty::UpvarId { var_id: var_id,
                                     closure_expr_id: fn_node_id };

        // Do we need to deref through an env reference?
        let has_env_deref = kind != ty::FnOnceUnboxedClosureKind;

        // Mutability of original variable itself
        let var_mutbl = MutabilityCategory::from_local(self.tcx(), var_id);

        // Mutability of environment dereference
        let env_mutbl = match kind {
            ty::FnOnceUnboxedClosureKind => var_mutbl,
            ty::FnMutUnboxedClosureKind => McInherited,
            ty::FnUnboxedClosureKind => McImmutable
        };

        // Look up the node ID of the closure body so we can construct
        // a free region within it
        let fn_body_id = {
            let fn_expr = match self.tcx().map.find(fn_node_id) {
                Some(ast_map::NodeExpr(e)) => e,
                _ => unreachable!()
            };

            match fn_expr.node {
                ast::ExprFnBlock(_, _, ref body) |
                ast::ExprProc(_, ref body) |
                ast::ExprUnboxedFn(_, _, _, ref body) => body.id,
                _ => unreachable!()
            }
        };

        // Region of environment pointer
        let env_region = ty::ReFree(ty::FreeRegion {
            scope_id: fn_body_id,
            bound_region: ty::BrEnv
        });

        let env_ptr = BorrowedPtr(if env_mutbl.is_mutable() {
            ty::MutBorrow
        } else {
            ty::ImmBorrow
        }, env_region);

        let var_ty = if_ok!(self.node_ty(var_id));

        // First, switch by capture mode
        Ok(match mode {
            ast::CaptureByValue => {
                let mut base = cmt_ {
                    id: id,
                    span: span,
                    cat: cat_upvar(Upvar {
                        id: upvar_id,
                        kind: kind,
                        is_unboxed: is_unboxed
                    }),
                    mutbl: var_mutbl,
                    ty: var_ty,
                    note: NoteNone
                };

                if has_env_deref {
                    // We need to add the env deref.  This means that
                    // the above is actually immutable and has a ref
                    // type.  However, nothing should actually look at
                    // the type, so we can get away with stuffing a
                    // `ty_err` in there instead of bothering to
                    // construct a proper one.
                    base.mutbl = McImmutable;
                    base.ty = ty::mk_err();
                    Rc::new(cmt_ {
                        id: id,
                        span: span,
                        cat: cat_deref(Rc::new(base), 0, env_ptr),
                        mutbl: env_mutbl,
                        ty: var_ty,
                        note: NoteClosureEnv(upvar_id)
                    })
                } else {
                    Rc::new(base)
                }
            },
            ast::CaptureByRef => {
                // The type here is actually a ref (or ref of a ref),
                // but we can again get away with not constructing one
                // properly since it will never be used.
                let mut base = cmt_ {
                    id: id,
                    span: span,
                    cat: cat_upvar(Upvar {
                        id: upvar_id,
                        kind: kind,
                        is_unboxed: is_unboxed
                    }),
                    mutbl: McImmutable,
                    ty: ty::mk_err(),
                    note: NoteNone
                };

                // As in the by-value case, add env deref if needed
                if has_env_deref {
                    base = cmt_ {
                        id: id,
                        span: span,
                        cat: cat_deref(Rc::new(base), 0, env_ptr),
                        mutbl: env_mutbl,
                        ty: ty::mk_err(),
                        note: NoteClosureEnv(upvar_id)
                    };
                }

                // Look up upvar borrow so we can get its region
                let upvar_borrow = self.typer.upvar_borrow(upvar_id);
                let ptr = BorrowedPtr(upvar_borrow.kind, upvar_borrow.region);

                Rc::new(cmt_ {
                    id: id,
                    span: span,
                    cat: cat_deref(Rc::new(base), 0, ptr),
                    mutbl: MutabilityCategory::from_borrow_kind(upvar_borrow.kind),
                    ty: var_ty,
                    note: NoteUpvarRef(upvar_id)
                })
            }
        })
    }

    pub fn cat_rvalue_node(&self,
                           id: ast::NodeId,
                           span: Span,
                           expr_ty: ty::t)
                           -> cmt {
        match self.typer.temporary_scope(id) {
            Some(scope) => {
                match ty::get(expr_ty).sty {
                    ty::ty_vec(_, Some(0)) => self.cat_rvalue(id, span, ty::ReStatic, expr_ty),
                    _ => self.cat_rvalue(id, span, ty::ReScope(scope), expr_ty)
                }
            }
            None => {
                self.cat_rvalue(id, span, ty::ReStatic, expr_ty)
            }
        }
    }

    pub fn cat_rvalue(&self,
                      cmt_id: ast::NodeId,
                      span: Span,
                      temp_scope: ty::Region,
                      expr_ty: ty::t) -> cmt {
        Rc::new(cmt_ {
            id:cmt_id,
            span:span,
            cat:cat_rvalue(temp_scope),
            mutbl:McDeclared,
            ty:expr_ty,
            note: NoteNone
        })
    }

    pub fn cat_field<N:ast_node>(&self,
                                 node: &N,
                                 base_cmt: cmt,
                                 f_name: ast::Ident,
                                 f_ty: ty::t)
                                 -> cmt {
        Rc::new(cmt_ {
            id: node.id(),
            span: node.span(),
            mutbl: base_cmt.mutbl.inherit(),
            cat: cat_interior(base_cmt, InteriorField(NamedField(f_name.name))),
            ty: f_ty,
            note: NoteNone
        })
    }

    pub fn cat_tup_field<N:ast_node>(&self,
                                     node: &N,
                                     base_cmt: cmt,
                                     f_idx: uint,
                                     f_ty: ty::t)
                                     -> cmt {
        Rc::new(cmt_ {
            id: node.id(),
            span: node.span(),
            mutbl: base_cmt.mutbl.inherit(),
            cat: cat_interior(base_cmt, InteriorField(PositionalField(f_idx))),
            ty: f_ty,
            note: NoteNone
        })
    }

    fn cat_deref<N:ast_node>(&self,
                             node: &N,
                             base_cmt: cmt,
                             deref_cnt: uint,
                             implicit: bool)
                             -> cmt {
        let adjustment = match self.typer.adjustments().borrow().find(&node.id()) {
            Some(adj) if ty::adjust_is_object(adj) => typeck::AutoObject,
            _ if deref_cnt != 0 => typeck::AutoDeref(deref_cnt),
            _ => typeck::NoAdjustment
        };

        let method_call = typeck::MethodCall {
            expr_id: node.id(),
            adjustment: adjustment
        };
        let method_ty = self.typer.node_method_ty(method_call);

        debug!("cat_deref: method_call={} method_ty={}",
            method_call, method_ty.map(|ty| ty.repr(self.tcx())));

        let base_cmt = match method_ty {
            Some(method_ty) => {
                let ref_ty = ty::ty_fn_ret(method_ty);
                self.cat_rvalue_node(node.id(), node.span(), ref_ty)
            }
            None => base_cmt
        };
        match ty::deref(base_cmt.ty, true) {
            Some(mt) => self.cat_deref_common(node, base_cmt, deref_cnt, mt.ty, implicit),
            None => {
                self.tcx().sess.span_bug(
                    node.span(),
                    format!("Explicit deref of non-derefable type: {}",
                            base_cmt.ty.repr(self.tcx())).as_slice());
            }
        }
    }

    fn cat_deref_common<N:ast_node>(&self,
                                    node: &N,
                                    base_cmt: cmt,
                                    deref_cnt: uint,
                                    deref_ty: ty::t,
                                    implicit: bool)
                                    -> cmt {
        let (m, cat) = match deref_kind(self.tcx(), base_cmt.ty) {
            deref_ptr(ptr) => {
                let ptr = if implicit {
                    match ptr {
                        BorrowedPtr(bk, r) => Implicit(bk, r),
                        _ => self.tcx().sess.span_bug(node.span(),
                            "Implicit deref of non-borrowed pointer")
                    }
                } else {
                    ptr
                };
                // for unique ptrs, we inherit mutability from the
                // owning reference.
                (MutabilityCategory::from_pointer_kind(base_cmt.mutbl, ptr),
                 cat_deref(base_cmt, deref_cnt, ptr))
            }
            deref_interior(interior) => {
                (base_cmt.mutbl.inherit(), cat_interior(base_cmt, interior))
            }
        };
        Rc::new(cmt_ {
            id: node.id(),
            span: node.span(),
            cat: cat,
            mutbl: m,
            ty: deref_ty,
            note: NoteNone
        })
    }

    pub fn cat_index<N:ast_node>(&self,
                                 elt: &N,
                                 mut base_cmt: cmt)
                                 -> cmt {
        //! Creates a cmt for an indexing operation (`[]`).
        //!
        //! One subtle aspect of indexing that may not be
        //! immediately obvious: for anything other than a fixed-length
        //! vector, an operation like `x[y]` actually consists of two
        //! disjoint (from the point of view of borrowck) operations.
        //! The first is a deref of `x` to create a pointer `p` that points
        //! at the first element in the array. The second operation is
        //! an index which adds `y*sizeof(T)` to `p` to obtain the
        //! pointer to `x[y]`. `cat_index` will produce a resulting
        //! cmt containing both this deref and the indexing,
        //! presuming that `base_cmt` is not of fixed-length type.
        //!
        //! # Parameters
        //! - `elt`: the AST node being indexed
        //! - `base_cmt`: the cmt of `elt`

        let method_call = typeck::MethodCall::expr(elt.id());
        let method_ty = self.typer.node_method_ty(method_call);

        let element_ty = match method_ty {
            Some(method_ty) => {
                let ref_ty = ty::ty_fn_ret(method_ty);
                base_cmt = self.cat_rvalue_node(elt.id(), elt.span(), ref_ty);
                *ty::ty_fn_args(method_ty).get(0)
            }
            None => {
                match ty::array_element_ty(base_cmt.ty) {
                    Some(ty) => ty,
                    None => {
                        self.tcx().sess.span_bug(
                            elt.span(),
                            format!("Explicit index of non-index type `{}`",
                                    base_cmt.ty.repr(self.tcx())).as_slice());
                    }
                }
            }
        };

        let m = base_cmt.mutbl.inherit();
        return interior(elt, base_cmt.clone(), base_cmt.ty, m, element_ty);

        fn interior<N: ast_node>(elt: &N,
                                 of_cmt: cmt,
                                 vec_ty: ty::t,
                                 mutbl: MutabilityCategory,
                                 element_ty: ty::t) -> cmt
        {
            Rc::new(cmt_ {
                id:elt.id(),
                span:elt.span(),
                cat:cat_interior(of_cmt, InteriorElement(element_kind(vec_ty))),
                mutbl:mutbl,
                ty:element_ty,
                note: NoteNone
            })
        }
    }

    // Takes either a vec or a reference to a vec and returns the cmt for the
    // underlying vec.
    fn deref_vec<N:ast_node>(&self,
                             elt: &N,
                             base_cmt: cmt)
                             -> cmt {
        match deref_kind(self.tcx(), base_cmt.ty) {
            deref_ptr(ptr) => {
                // for unique ptrs, we inherit mutability from the
                // owning reference.
                let m = MutabilityCategory::from_pointer_kind(base_cmt.mutbl, ptr);

                // the deref is explicit in the resulting cmt
                Rc::new(cmt_ {
                    id:elt.id(),
                    span:elt.span(),
                    cat:cat_deref(base_cmt.clone(), 0, ptr),
                    mutbl:m,
                    ty: match ty::deref(base_cmt.ty, false) {
                        Some(mt) => mt.ty,
                        None => self.tcx().sess.bug("Found non-derefable type")
                    },
                    note: NoteNone
                })
            }

            deref_interior(_) => {
                base_cmt
            }
        }
    }

    pub fn cat_slice_pattern(&self,
                             vec_cmt: cmt,
                             slice_pat: &ast::Pat)
                             -> McResult<(cmt, ast::Mutability, ty::Region)> {
        /*!
         * Given a pattern P like: `[_, ..Q, _]`, where `vec_cmt` is
         * the cmt for `P`, `slice_pat` is the pattern `Q`, returns:
         * - a cmt for `Q`
         * - the mutability and region of the slice `Q`
         *
         * These last two bits of info happen to be things that
         * borrowck needs.
         */

        let slice_ty = if_ok!(self.node_ty(slice_pat.id));
        let (slice_mutbl, slice_r) = vec_slice_info(self.tcx(),
                                                    slice_pat,
                                                    slice_ty);
        let cmt_slice = self.cat_index(slice_pat, self.deref_vec(slice_pat, vec_cmt));
        return Ok((cmt_slice, slice_mutbl, slice_r));

        fn vec_slice_info(tcx: &ty::ctxt,
                          pat: &ast::Pat,
                          slice_ty: ty::t)
                          -> (ast::Mutability, ty::Region) {
            /*!
             * In a pattern like [a, b, ..c], normally `c` has slice type,
             * but if you have [a, b, ..ref c], then the type of `ref c`
             * will be `&&[]`, so to extract the slice details we have
             * to recurse through rptrs.
             */

            match ty::get(slice_ty).sty {
                ty::ty_rptr(r, ref mt) => match ty::get(mt.ty).sty {
                    ty::ty_vec(_, None) => (mt.mutbl, r),
                    _ => vec_slice_info(tcx, pat, mt.ty),
                },

                _ => {
                    tcx.sess.span_bug(pat.span,
                                      "type of slice pattern is not a slice");
                }
            }
        }
    }

    pub fn cat_imm_interior<N:ast_node>(&self,
                                        node: &N,
                                        base_cmt: cmt,
                                        interior_ty: ty::t,
                                        interior: InteriorKind)
                                        -> cmt {
        Rc::new(cmt_ {
            id: node.id(),
            span: node.span(),
            mutbl: base_cmt.mutbl.inherit(),
            cat: cat_interior(base_cmt, interior),
            ty: interior_ty,
            note: NoteNone
        })
    }

    pub fn cat_downcast<N:ast_node>(&self,
                                    node: &N,
                                    base_cmt: cmt,
                                    downcast_ty: ty::t)
                                    -> cmt {
        Rc::new(cmt_ {
            id: node.id(),
            span: node.span(),
            mutbl: base_cmt.mutbl.inherit(),
            cat: cat_downcast(base_cmt),
            ty: downcast_ty,
            note: NoteNone
        })
    }

    pub fn cat_pattern(&self,
                       cmt: cmt,
                       pat: &ast::Pat,
                       op: |&MemCategorizationContext<TYPER>,
                            cmt,
                            &ast::Pat|)
                       -> McResult<()> {
        // Here, `cmt` is the categorization for the value being
        // matched and pat is the pattern it is being matched against.
        //
        // In general, the way that this works is that we walk down
        // the pattern, constructing a cmt that represents the path
        // that will be taken to reach the value being matched.
        //
        // When we encounter named bindings, we take the cmt that has
        // been built up and pass it off to guarantee_valid() so that
        // we can be sure that the binding will remain valid for the
        // duration of the arm.
        //
        // (*2) There is subtlety concerning the correspondence between
        // pattern ids and types as compared to *expression* ids and
        // types. This is explained briefly. on the definition of the
        // type `cmt`, so go off and read what it says there, then
        // come back and I'll dive into a bit more detail here. :) OK,
        // back?
        //
        // In general, the id of the cmt should be the node that
        // "produces" the value---patterns aren't executable code
        // exactly, but I consider them to "execute" when they match a
        // value, and I consider them to produce the value that was
        // matched. So if you have something like:
        //
        //     let x = @@3;
        //     match x {
        //       @@y { ... }
        //     }
        //
        // In this case, the cmt and the relevant ids would be:
        //
        //     CMT             Id                  Type of Id Type of cmt
        //
        //     local(x)->@->@
        //     ^~~~~~~^        `x` from discr      @@int      @@int
        //     ^~~~~~~~~~^     `@@y` pattern node  @@int      @int
        //     ^~~~~~~~~~~~~^  `@y` pattern node   @int       int
        //
        // You can see that the types of the id and the cmt are in
        // sync in the first line, because that id is actually the id
        // of an expression. But once we get to pattern ids, the types
        // step out of sync again. So you'll see below that we always
        // get the type of the *subpattern* and use that.

        debug!("cat_pattern: id={} pat={} cmt={}",
               pat.id, pprust::pat_to_string(pat),
               cmt.repr(self.tcx()));

        op(self, cmt.clone(), pat);

        match pat.node {
          ast::PatWild(_) => {
            // _
          }

          ast::PatEnum(_, None) => {
            // variant(..)
          }
          ast::PatEnum(_, Some(ref subpats)) => {
            match self.tcx().def_map.borrow().find(&pat.id) {
                Some(&def::DefVariant(enum_did, _, _)) => {
                    // variant(x, y, z)

                    let downcast_cmt = {
                        if ty::enum_is_univariant(self.tcx(), enum_did) {
                            cmt // univariant, no downcast needed
                        } else {
                            self.cat_downcast(pat, cmt.clone(), cmt.ty)
                        }
                    };

                    for (i, subpat) in subpats.iter().enumerate() {
                        let subpat_ty = if_ok!(self.pat_ty(&**subpat)); // see (*2)

                        let subcmt =
                            self.cat_imm_interior(
                                pat, downcast_cmt.clone(), subpat_ty,
                                InteriorField(PositionalField(i)));

                        if_ok!(self.cat_pattern(subcmt, &**subpat, |x,y,z| op(x,y,z)));
                    }
                }
                Some(&def::DefStruct(..)) => {
                    for (i, subpat) in subpats.iter().enumerate() {
                        let subpat_ty = if_ok!(self.pat_ty(&**subpat)); // see (*2)
                        let cmt_field =
                            self.cat_imm_interior(
                                pat, cmt.clone(), subpat_ty,
                                InteriorField(PositionalField(i)));
                        if_ok!(self.cat_pattern(cmt_field, &**subpat,
                                                |x,y,z| op(x,y,z)));
                    }
                }
                Some(&def::DefConst(..)) => {
                    for subpat in subpats.iter() {
                        if_ok!(self.cat_pattern(cmt.clone(), &**subpat, |x,y,z| op(x,y,z)));
                    }
                }
                _ => {
                    self.tcx().sess.span_bug(
                        pat.span,
                        "enum pattern didn't resolve to enum or struct");
                }
            }
          }

          ast::PatIdent(_, _, Some(ref subpat)) => {
              if_ok!(self.cat_pattern(cmt, &**subpat, op));
          }

          ast::PatIdent(_, _, None) => {
              // nullary variant or identifier: ignore
          }

          ast::PatStruct(_, ref field_pats, _) => {
            // {f1: p1, ..., fN: pN}
            for fp in field_pats.iter() {
                let field_ty = if_ok!(self.pat_ty(&*fp.pat)); // see (*2)
                let cmt_field = self.cat_field(pat, cmt.clone(), fp.ident, field_ty);
                if_ok!(self.cat_pattern(cmt_field, &*fp.pat, |x,y,z| op(x,y,z)));
            }
          }

          ast::PatTup(ref subpats) => {
            // (p1, ..., pN)
            for (i, subpat) in subpats.iter().enumerate() {
                let subpat_ty = if_ok!(self.pat_ty(&**subpat)); // see (*2)
                let subcmt =
                    self.cat_imm_interior(
                        pat, cmt.clone(), subpat_ty,
                        InteriorField(PositionalField(i)));
                if_ok!(self.cat_pattern(subcmt, &**subpat, |x,y,z| op(x,y,z)));
            }
          }

          ast::PatBox(ref subpat) | ast::PatRegion(ref subpat) => {
            // @p1, ~p1, ref p1
            let subcmt = self.cat_deref(pat, cmt, 0, false);
            if_ok!(self.cat_pattern(subcmt, &**subpat, op));
          }

          ast::PatVec(ref before, ref slice, ref after) => {
              let elt_cmt = self.cat_index(pat, self.deref_vec(pat, cmt));
              for before_pat in before.iter() {
                  if_ok!(self.cat_pattern(elt_cmt.clone(), &**before_pat,
                                          |x,y,z| op(x,y,z)));
              }
              for slice_pat in slice.iter() {
                  let slice_ty = if_ok!(self.pat_ty(&**slice_pat));
                  let slice_cmt = self.cat_rvalue_node(pat.id(), pat.span(), slice_ty);
                  if_ok!(self.cat_pattern(slice_cmt, &**slice_pat, |x,y,z| op(x,y,z)));
              }
              for after_pat in after.iter() {
                  if_ok!(self.cat_pattern(elt_cmt.clone(), &**after_pat, |x,y,z| op(x,y,z)));
              }
          }

          ast::PatLit(_) | ast::PatRange(_, _) => {
              /*always ok*/
          }

          ast::PatMac(_) => {
              self.tcx().sess.span_bug(pat.span, "unexpanded macro");
          }
        }

        Ok(())
    }

    pub fn cmt_to_string(&self, cmt: &cmt_) -> String {
        fn upvar_to_string(upvar: &Upvar, is_copy: bool) -> String {
            if upvar.is_unboxed {
                let kind = match upvar.kind {
                    ty::FnUnboxedClosureKind => "Fn",
                    ty::FnMutUnboxedClosureKind => "FnMut",
                    ty::FnOnceUnboxedClosureKind => "FnOnce"
                };
                format!("captured outer variable in an `{}` closure", kind)
            } else {
                (match (upvar.kind, is_copy) {
                    (ty::FnOnceUnboxedClosureKind, true) => "captured outer variable in a proc",
                    _ => "captured outer variable"
                }).to_string()
            }
        }

        match cmt.cat {
          cat_static_item => {
              "static item".to_string()
          }
          cat_rvalue(..) => {
              "non-lvalue".to_string()
          }
          cat_local(vid) => {
              match self.tcx().map.find(vid) {
                  Some(ast_map::NodeArg(_)) => {
                      "argument".to_string()
                  }
                  _ => "local variable".to_string()
              }
          }
          cat_deref(_, _, pk) => {
              let upvar = cmt.upvar();
              match upvar.as_ref().map(|i| &i.cat) {
                  Some(&cat_upvar(ref var)) => {
                      upvar_to_string(var, false)
                  }
                  Some(_) => unreachable!(),
                  None => {
                      match pk {
                          Implicit(..) => {
                            "dereference (dereference is implicit, due to indexing)".to_string()
                          }
                          OwnedPtr => format!("dereference of `{}`", ptr_sigil(pk)),
                          _ => format!("dereference of `{}`-pointer", ptr_sigil(pk))
                      }
                  }
              }
          }
          cat_interior(_, InteriorField(NamedField(_))) => {
              "field".to_string()
          }
          cat_interior(_, InteriorField(PositionalField(_))) => {
              "anonymous field".to_string()
          }
          cat_interior(_, InteriorElement(VecElement)) => {
              "vec content".to_string()
          }
          cat_interior(_, InteriorElement(OtherElement)) => {
              "indexed content".to_string()
          }
          cat_upvar(ref var) => {
              upvar_to_string(var, true)
          }
          cat_discr(ref cmt, _) => {
            self.cmt_to_string(&**cmt)
          }
          cat_downcast(ref cmt) => {
            self.cmt_to_string(&**cmt)
          }
        }
    }
}

pub enum InteriorSafety {
    InteriorUnsafe,
    InteriorSafe
}

pub enum AliasableReason {
    AliasableBorrowed,
    AliasableClosure(ast::NodeId), // Aliasable due to capture Fn closure env
    AliasableOther,
    AliasableStatic(InteriorSafety),
    AliasableStaticMut(InteriorSafety),
}

impl cmt_ {
    pub fn guarantor(&self) -> cmt {
        //! Returns `self` after stripping away any owned pointer derefs or
        //! interior content. The return value is basically the `cmt` which
        //! determines how long the value in `self` remains live.

        match self.cat {
            cat_rvalue(..) |
            cat_static_item |
            cat_local(..) |
            cat_deref(_, _, UnsafePtr(..)) |
            cat_deref(_, _, BorrowedPtr(..)) |
            cat_deref(_, _, Implicit(..)) |
            cat_upvar(..) => {
                Rc::new((*self).clone())
            }
            cat_downcast(ref b) |
            cat_discr(ref b, _) |
            cat_interior(ref b, _) |
            cat_deref(ref b, _, OwnedPtr) => {
                b.guarantor()
            }
        }
    }

    pub fn freely_aliasable(&self, ctxt: &ty::ctxt) -> Option<AliasableReason> {
        /*!
         * Returns `Some(_)` if this lvalue represents a freely aliasable
         * pointer type.
         */

        // Maybe non-obvious: copied upvars can only be considered
        // non-aliasable in once closures, since any other kind can be
        // aliased and eventually recused.

        match self.cat {
            cat_deref(ref b, _, BorrowedPtr(ty::MutBorrow, _)) |
            cat_deref(ref b, _, Implicit(ty::MutBorrow, _)) |
            cat_deref(ref b, _, BorrowedPtr(ty::UniqueImmBorrow, _)) |
            cat_deref(ref b, _, Implicit(ty::UniqueImmBorrow, _)) |
            cat_downcast(ref b) |
            cat_deref(ref b, _, OwnedPtr) |
            cat_interior(ref b, _) |
            cat_discr(ref b, _) => {
                // Aliasability depends on base cmt
                b.freely_aliasable(ctxt)
            }

            cat_rvalue(..) |
            cat_local(..) |
            cat_upvar(..) |
            cat_deref(_, _, UnsafePtr(..)) => { // yes, it's aliasable, but...
                None
            }

            cat_static_item(..) => {
                let int_safe = if ty::type_interior_is_unsafe(ctxt, self.ty) {
                    InteriorUnsafe
                } else {
                    InteriorSafe
                };

                if self.mutbl.is_mutable() {
                    Some(AliasableStaticMut(int_safe))
                } else {
                    Some(AliasableStatic(int_safe))
                }
            }

            cat_deref(ref base, _, BorrowedPtr(ty::ImmBorrow, _)) |
            cat_deref(ref base, _, Implicit(ty::ImmBorrow, _)) => {
                match base.cat {
                    cat_upvar(Upvar{ id, .. }) => Some(AliasableClosure(id.closure_expr_id)),
                    _ => Some(AliasableBorrowed)
                }
            }
        }
    }

    // Digs down through one or two layers of deref and grabs the cmt
    // for the upvar if a note indicates there is one.
    pub fn upvar(&self) -> Option<cmt> {
        match self.note {
            NoteClosureEnv(..) | NoteUpvarRef(..) => {
                Some(match self.cat {
                    cat_deref(ref inner, _, _) => {
                        match inner.cat {
                            cat_deref(ref inner, _, _) => inner.clone(),
                            cat_upvar(..) => inner.clone(),
                            _ => unreachable!()
                        }
                    }
                    _ => unreachable!()
                })
            }
            NoteNone => None
        }
    }
}

impl Repr for cmt_ {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        format!("{{{} id:{} m:{} ty:{}}}",
                self.cat.repr(tcx),
                self.id,
                self.mutbl,
                self.ty.repr(tcx))
    }
}

impl Repr for categorization {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        match *self {
            cat_static_item |
            cat_rvalue(..) |
            cat_local(..) |
            cat_upvar(..) => {
                format!("{}", *self)
            }
            cat_deref(ref cmt, derefs, ptr) => {
                format!("{}-{}{}->", cmt.cat.repr(tcx), ptr_sigil(ptr), derefs)
            }
            cat_interior(ref cmt, interior) => {
                format!("{}.{}", cmt.cat.repr(tcx), interior.repr(tcx))
            }
            cat_downcast(ref cmt) => {
                format!("{}->(enum)", cmt.cat.repr(tcx))
            }
            cat_discr(ref cmt, _) => {
                cmt.cat.repr(tcx)
            }
        }
    }
}

pub fn ptr_sigil(ptr: PointerKind) -> &'static str {
    match ptr {
        OwnedPtr => "Box",
        BorrowedPtr(ty::ImmBorrow, _) |
        Implicit(ty::ImmBorrow, _) => "&",
        BorrowedPtr(ty::MutBorrow, _) |
        Implicit(ty::MutBorrow, _) => "&mut",
        BorrowedPtr(ty::UniqueImmBorrow, _) |
        Implicit(ty::UniqueImmBorrow, _) => "&unique",
        UnsafePtr(_) => "*"
    }
}

impl Repr for InteriorKind {
    fn repr(&self, _tcx: &ty::ctxt) -> String {
        match *self {
            InteriorField(NamedField(fld)) => {
                token::get_name(fld).get().to_string()
            }
            InteriorField(PositionalField(i)) => format!("#{}", i),
            InteriorElement(_) => "[]".to_string(),
        }
    }
}

fn element_kind(t: ty::t) -> ElementKind {
    match ty::get(t).sty {
        ty::ty_rptr(_, ty::mt{ty:ty, ..}) |
        ty::ty_uniq(ty) => match ty::get(ty).sty {
            ty::ty_vec(_, None) => VecElement,
            _ => OtherElement
        },
        ty::ty_vec(..) => VecElement,
        _ => OtherElement
    }
}
