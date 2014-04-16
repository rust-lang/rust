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
 * Now, cat_expr() classies the expression Expr and the address A=ToAddr(Expr)
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
 * decomposed into two operations: a derefence to reach the array data and
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

use middle::ty;
use middle::typeck;
use util::ppaux::{ty_to_str, Repr};

use syntax::ast::{MutImmutable, MutMutable};
use syntax::ast;
use syntax::codemap::Span;
use syntax::print::pprust;
use syntax::parse::token;

#[deriving(Eq)]
pub enum categorization {
    cat_rvalue(ty::Region),            // temporary val, argument is its scope
    cat_static_item,
    cat_copied_upvar(CopiedUpvar),     // upvar copied into proc env
    cat_upvar(ty::UpvarId, ty::UpvarBorrow), // by ref upvar from stack closure
    cat_local(ast::NodeId),            // local variable
    cat_arg(ast::NodeId),              // formal argument
    cat_deref(cmt, uint, PointerKind), // deref of a ptr
    cat_interior(cmt, InteriorKind),   // something interior: field, tuple, etc
    cat_downcast(cmt),                 // selects a particular enum variant (*1)
    cat_discr(cmt, ast::NodeId),       // match discriminant (see preserve())

    // (*1) downcast is only required if the enum has more than one variant
}

#[deriving(Eq)]
pub struct CopiedUpvar {
    pub upvar_id: ast::NodeId,
    pub onceness: ast::Onceness,
}

// different kinds of pointers:
#[deriving(Eq, TotalEq, Hash)]
pub enum PointerKind {
    OwnedPtr,
    GcPtr,
    BorrowedPtr(ty::BorrowKind, ty::Region),
    UnsafePtr(ast::Mutability),
}

// We use the term "interior" to mean "something reachable from the
// base without a pointer dereference", e.g. a field
#[deriving(Eq, TotalEq, Hash)]
pub enum InteriorKind {
    InteriorField(FieldName),
    InteriorElement(ElementKind),
}

#[deriving(Eq, TotalEq, Hash)]
pub enum FieldName {
    NamedField(ast::Name),
    PositionalField(uint)
}

#[deriving(Eq, TotalEq, Hash)]
pub enum ElementKind {
    VecElement,
    StrElement,
    OtherElement,
}

#[deriving(Eq, TotalEq, Hash, Show)]
pub enum MutabilityCategory {
    McImmutable, // Immutable.
    McDeclared,  // Directly declared as mutable.
    McInherited, // Inherited from the fact that owner is mutable.
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
// (`@T`). So use `cmt.type` to find the type of the value in a consistent
// fashion. For more details, see the method `cat_pattern`
#[deriving(Eq)]
pub struct cmt_ {
    pub id: ast::NodeId,          // id of expr/pat producing this value
    pub span: Span,                // span of same expr/pat
    pub cat: categorization,       // categorization of expr
    pub mutbl: MutabilityCategory, // mutability of expr as lvalue
    pub ty: ty::t                  // type of the expr (*see WARNING above*)
}

pub type cmt = @cmt_;

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
        ty::ty_trait(~ty::TyTrait { store: ty::UniqTraitStore, .. }) |
        ty::ty_vec(_, ty::VstoreUniq) |
        ty::ty_str(ty::VstoreUniq) |
        ty::ty_closure(~ty::ClosureTy {store: ty::UniqTraitStore, ..}) => {
            Some(deref_ptr(OwnedPtr))
        }

        ty::ty_rptr(r, mt) => {
            let kind = ty::BorrowKind::from_mutbl(mt.mutbl);
            Some(deref_ptr(BorrowedPtr(kind, r)))
        }
        ty::ty_vec(_, ty::VstoreSlice(r, mutbl)) |
        ty::ty_trait(~ty::TyTrait { store: ty::RegionTraitStore(r, mutbl), .. }) => {
            let kind = ty::BorrowKind::from_mutbl(mutbl);
            Some(deref_ptr(BorrowedPtr(kind, r)))
        }

        ty::ty_str(ty::VstoreSlice(r, ())) |
        ty::ty_closure(~ty::ClosureTy {store: ty::RegionTraitStore(r, _), ..}) => {
            Some(deref_ptr(BorrowedPtr(ty::ImmBorrow, r)))
        }

        ty::ty_box(..) => {
            Some(deref_ptr(GcPtr))
        }

        ty::ty_ptr(ref mt) => {
            Some(deref_ptr(UnsafePtr(mt.mutbl)))
        }

        ty::ty_enum(..) |
        ty::ty_struct(..) => { // newtype
            Some(deref_interior(InteriorField(PositionalField(0))))
        }

        ty::ty_vec(_, ty::VstoreFixed(_)) |
        ty::ty_str(ty::VstoreFixed(_)) => {
            Some(deref_interior(InteriorElement(element_kind(t))))
        }

        _ => None
    }
}

pub fn deref_kind(tcx: &ty::ctxt, t: ty::t) -> deref_kind {
    match opt_deref_kind(t) {
      Some(k) => k,
      None => {
        tcx.sess.bug(
            format!("deref_cat() invoked on non-derefable type {}",
                 ty_to_str(tcx, t)));
      }
    }
}

trait ast_node {
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

pub struct MemCategorizationContext<TYPER> {
    pub typer: TYPER
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
pub trait Typer {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt;
    fn node_ty(&mut self, id: ast::NodeId) -> McResult<ty::t>;
    fn node_method_ty(&self, method_call: typeck::MethodCall) -> Option<ty::t>;
    fn adjustment(&mut self, node_id: ast::NodeId) -> Option<@ty::AutoAdjustment>;
    fn is_method_call(&mut self, id: ast::NodeId) -> bool;
    fn temporary_scope(&mut self, rvalue_id: ast::NodeId) -> Option<ast::NodeId>;
    fn upvar_borrow(&mut self, upvar_id: ty::UpvarId) -> ty::UpvarBorrow;
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
            BorrowedPtr(borrow_kind, _) => {
                MutabilityCategory::from_borrow_kind(borrow_kind)
            }
            GcPtr => {
                McImmutable
            }
            UnsafePtr(m) => {
                MutabilityCategory::from_mutbl(m)
            }
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

impl<TYPER:Typer> MemCategorizationContext<TYPER> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt {
        self.typer.tcx()
    }

    fn adjustment(&mut self, id: ast::NodeId) -> Option<@ty::AutoAdjustment> {
        self.typer.adjustment(id)
    }

    fn expr_ty(&mut self, expr: &ast::Expr) -> McResult<ty::t> {
        self.typer.node_ty(expr.id)
    }

    fn expr_ty_adjusted(&mut self, expr: &ast::Expr) -> McResult<ty::t> {
        let unadjusted_ty = if_ok!(self.expr_ty(expr));
        let adjustment = self.adjustment(expr.id);
        Ok(ty::adjust_ty(self.tcx(), expr.span, expr.id, unadjusted_ty, adjustment,
                         |method_call| self.typer.node_method_ty(method_call)))
    }

    fn node_ty(&mut self, id: ast::NodeId) -> McResult<ty::t> {
        self.typer.node_ty(id)
    }

    fn pat_ty(&mut self, pat: @ast::Pat) -> McResult<ty::t> {
        self.typer.node_ty(pat.id)
    }

    pub fn cat_expr(&mut self, expr: &ast::Expr) -> McResult<cmt> {
        match self.adjustment(expr.id) {
            None => {
                // No adjustments.
                self.cat_expr_unadjusted(expr)
            }

            Some(adjustment) => {
                match *adjustment {
                    ty::AutoObject(..) => {
                        // Implicity casts a concrete object to trait object
                        // so just patch up the type
                        let expr_ty = if_ok!(self.expr_ty_adjusted(expr));
                        let expr_cmt = if_ok!(self.cat_expr_unadjusted(expr));
                        Ok(@cmt_ {ty: expr_ty, ..*expr_cmt})
                    }

                    ty::AutoAddEnv(..) => {
                        // Convert a bare fn to a closure by adding NULL env.
                        // Result is an rvalue.
                        let expr_ty = if_ok!(self.expr_ty_adjusted(expr));
                        Ok(self.cat_rvalue_node(expr.id(), expr.span(), expr_ty))
                    }

                    ty::AutoDerefRef(
                        ty::AutoDerefRef {
                            autoref: Some(_), ..}) => {
                        // Equivalent to &*expr or something similar.
                        // Result is an rvalue.
                        let expr_ty = if_ok!(self.expr_ty_adjusted(expr));
                        Ok(self.cat_rvalue_node(expr.id(), expr.span(), expr_ty))
                    }

                    ty::AutoDerefRef(
                        ty::AutoDerefRef {
                            autoref: None, autoderefs: autoderefs}) => {
                        // Equivalent to *expr or something similar.
                        self.cat_expr_autoderefd(expr, autoderefs)
                    }
                }
            }
        }
    }

    pub fn cat_expr_autoderefd(&mut self, expr: &ast::Expr, autoderefs: uint)
                               -> McResult<cmt> {
        let mut cmt = if_ok!(self.cat_expr_unadjusted(expr));
        for deref in range(1u, autoderefs + 1) {
            cmt = self.cat_deref(expr, cmt, deref);
        }
        return Ok(cmt);
    }

    pub fn cat_expr_unadjusted(&mut self, expr: &ast::Expr) -> McResult<cmt> {
        debug!("cat_expr: id={} expr={}", expr.id, expr.repr(self.tcx()));

        let expr_ty = if_ok!(self.expr_ty(expr));
        match expr.node {
          ast::ExprUnary(ast::UnDeref, e_base) => {
            let base_cmt = if_ok!(self.cat_expr(e_base));
            Ok(self.cat_deref(expr, base_cmt, 0))
          }

          ast::ExprField(base, f_name, _) => {
            let base_cmt = if_ok!(self.cat_expr(base));
            Ok(self.cat_field(expr, base_cmt, f_name, expr_ty))
          }

          ast::ExprIndex(base, _) => {
            if self.typer.is_method_call(expr.id) {
                return Ok(self.cat_rvalue_node(expr.id(), expr.span(), expr_ty));
            }

            let base_cmt = if_ok!(self.cat_expr(base));
            Ok(self.cat_index(expr, base_cmt, 0))
          }

          ast::ExprPath(_) => {
            let def = self.tcx().def_map.borrow().get_copy(&expr.id);
            self.cat_def(expr.id, expr.span, expr_ty, def)
          }

          ast::ExprParen(e) => self.cat_expr_unadjusted(e),

          ast::ExprAddrOf(..) | ast::ExprCall(..) |
          ast::ExprAssign(..) | ast::ExprAssignOp(..) |
          ast::ExprFnBlock(..) | ast::ExprProc(..) | ast::ExprRet(..) |
          ast::ExprUnary(..) |
          ast::ExprMethodCall(..) | ast::ExprCast(..) | ast::ExprVstore(..) |
          ast::ExprVec(..) | ast::ExprTup(..) | ast::ExprIf(..) |
          ast::ExprBinary(..) | ast::ExprWhile(..) |
          ast::ExprBlock(..) | ast::ExprLoop(..) | ast::ExprMatch(..) |
          ast::ExprLit(..) | ast::ExprBreak(..) | ast::ExprMac(..) |
          ast::ExprAgain(..) | ast::ExprStruct(..) | ast::ExprRepeat(..) |
          ast::ExprInlineAsm(..) | ast::ExprBox(..) => {
            Ok(self.cat_rvalue_node(expr.id(), expr.span(), expr_ty))
          }

          ast::ExprForLoop(..) => fail!("non-desugared expr_for_loop")
        }
    }

    pub fn cat_def(&mut self,
                   id: ast::NodeId,
                   span: Span,
                   expr_ty: ty::t,
                   def: ast::Def)
                   -> McResult<cmt> {
        debug!("cat_def: id={} expr={}",
               id, expr_ty.repr(self.tcx()));

        match def {
          ast::DefStruct(..) | ast::DefVariant(..) => {
                Ok(self.cat_rvalue_node(id, span, expr_ty))
          }
          ast::DefFn(..) | ast::DefStaticMethod(..) | ast::DefMod(_) |
          ast::DefForeignMod(_) | ast::DefStatic(_, false) |
          ast::DefUse(_) | ast::DefTrait(_) | ast::DefTy(_) | ast::DefPrimTy(_) |
          ast::DefTyParam(..) | ast::DefTyParamBinder(..) | ast::DefRegion(_) |
          ast::DefLabel(_) | ast::DefSelfTy(..) | ast::DefMethod(..) => {
              Ok(@cmt_ {
                  id:id,
                  span:span,
                  cat:cat_static_item,
                  mutbl: McImmutable,
                  ty:expr_ty
              })
          }

          ast::DefStatic(_, true) => {
              Ok(@cmt_ {
                  id:id,
                  span:span,
                  cat:cat_static_item,
                  mutbl: McDeclared,
                  ty:expr_ty
              })
          }

          ast::DefArg(vid, binding_mode) => {
            // Idea: make this could be rewritten to model by-ref
            // stuff as `&const` and `&mut`?

            // m: mutability of the argument
            let m = match binding_mode {
                ast::BindByValue(ast::MutMutable) => McDeclared,
                _ => McImmutable
            };
            Ok(@cmt_ {
                id: id,
                span: span,
                cat: cat_arg(vid),
                mutbl: m,
                ty:expr_ty
            })
          }

          ast::DefUpvar(var_id, _, fn_node_id, _) => {
              let ty = if_ok!(self.node_ty(fn_node_id));
              match ty::get(ty).sty {
                  ty::ty_closure(ref closure_ty) => {
                      // Decide whether to use implicit reference or by copy/move
                      // capture for the upvar. This, combined with the onceness,
                      // determines whether the closure can move out of it.
                      let var_is_refd = match (closure_ty.store, closure_ty.onceness) {
                          // Many-shot stack closures can never move out.
                          (ty::RegionTraitStore(..), ast::Many) => true,
                          // 1-shot stack closures can move out.
                          (ty::RegionTraitStore(..), ast::Once) => false,
                          // Heap closures always capture by copy/move, and can
                          // move out if they are once.
                          (ty::UniqTraitStore, _) => false,

                      };
                      if var_is_refd {
                          self.cat_upvar(id, span, var_id, fn_node_id)
                      } else {
                          // FIXME #2152 allow mutation of moved upvars
                          Ok(@cmt_ {
                              id:id,
                              span:span,
                              cat:cat_copied_upvar(CopiedUpvar {
                                  upvar_id: var_id,
                                  onceness: closure_ty.onceness}),
                              mutbl:McImmutable,
                              ty:expr_ty
                          })
                      }
                  }
                  _ => {
                      self.tcx().sess.span_bug(
                          span,
                          format!("Upvar of non-closure {} - {}",
                                  fn_node_id, ty.repr(self.tcx())));
                  }
              }
          }

          ast::DefLocal(vid, binding_mode) |
          ast::DefBinding(vid, binding_mode) => {
            // by-value/by-ref bindings are local variables
            let m = match binding_mode {
                ast::BindByValue(ast::MutMutable) => McDeclared,
                _ => McImmutable
            };

            Ok(@cmt_ {
                id: id,
                span: span,
                cat: cat_local(vid),
                mutbl: m,
                ty: expr_ty
            })
          }
        }
    }

    fn cat_upvar(&mut self,
                 id: ast::NodeId,
                 span: Span,
                 var_id: ast::NodeId,
                 fn_node_id: ast::NodeId)
                 -> McResult<cmt> {
        /*!
         * Upvars through a closure are in fact indirect
         * references. That is, when a closure refers to a
         * variable from a parent stack frame like `x = 10`,
         * that is equivalent to `*x_ = 10` where `x_` is a
         * borrowed pointer (`&mut x`) created when the closure
         * was created and store in the environment. This
         * equivalence is expose in the mem-categorization.
         */

        let upvar_id = ty::UpvarId { var_id: var_id,
                                     closure_expr_id: fn_node_id };

        let upvar_borrow = self.typer.upvar_borrow(upvar_id);

        let var_ty = if_ok!(self.node_ty(var_id));

        // We can't actually represent the types of all upvars
        // as user-describable types, since upvars support const
        // and unique-imm borrows! Therefore, we cheat, and just
        // give err type. Nobody should be inspecting this type anyhow.
        let upvar_ty = ty::mk_err();

        let base_cmt = @cmt_ {
            id:id,
            span:span,
            cat:cat_upvar(upvar_id, upvar_borrow),
            mutbl:McImmutable,
            ty:upvar_ty,
        };

        let ptr = BorrowedPtr(upvar_borrow.kind, upvar_borrow.region);

        let deref_cmt = @cmt_ {
            id:id,
            span:span,
            cat:cat_deref(base_cmt, 0, ptr),
            mutbl:MutabilityCategory::from_borrow_kind(upvar_borrow.kind),
            ty:var_ty,
        };

        Ok(deref_cmt)
    }

    pub fn cat_rvalue_node(&mut self,
                           id: ast::NodeId,
                           span: Span,
                           expr_ty: ty::t)
                           -> cmt {
        match self.typer.temporary_scope(id) {
            Some(scope) => {
                self.cat_rvalue(id, span, ty::ReScope(scope), expr_ty)
            }
            None => {
                self.cat_rvalue(id, span, ty::ReStatic, expr_ty)
            }
        }
    }

    pub fn cat_rvalue(&mut self,
                      cmt_id: ast::NodeId,
                      span: Span,
                      temp_scope: ty::Region,
                      expr_ty: ty::t) -> cmt {
        @cmt_ {
            id:cmt_id,
            span:span,
            cat:cat_rvalue(temp_scope),
            mutbl:McDeclared,
            ty:expr_ty
        }
    }

    pub fn cat_field<N:ast_node>(&mut self,
                                 node: &N,
                                 base_cmt: cmt,
                                 f_name: ast::Ident,
                                 f_ty: ty::t)
                                 -> cmt {
        @cmt_ {
            id: node.id(),
            span: node.span(),
            cat: cat_interior(base_cmt, InteriorField(NamedField(f_name.name))),
            mutbl: base_cmt.mutbl.inherit(),
            ty: f_ty
        }
    }

    pub fn cat_deref_obj<N:ast_node>(&mut self, node: &N, base_cmt: cmt) -> cmt {
        self.cat_deref_common(node, base_cmt, 0, ty::mk_nil())
    }

    fn cat_deref<N:ast_node>(&mut self,
                             node: &N,
                             base_cmt: cmt,
                             deref_cnt: uint)
                             -> cmt {
        let method_call = typeck::MethodCall {
            expr_id: node.id(),
            autoderef: deref_cnt as u32
        };
        let method_ty = self.typer.node_method_ty(method_call);

        debug!("cat_deref: method_call={:?} method_ty={}",
            method_call, method_ty.map(|ty| ty.repr(self.tcx())));

        let base_cmt = match method_ty {
            Some(method_ty) => {
                let ref_ty = ty::ty_fn_ret(method_ty);
                self.cat_rvalue_node(node.id(), node.span(), ref_ty)
            }
            None => base_cmt
        };
        match ty::deref(base_cmt.ty, true) {
            Some(mt) => self.cat_deref_common(node, base_cmt, deref_cnt, mt.ty),
            None => {
                self.tcx().sess.span_bug(
                    node.span(),
                    format!("Explicit deref of non-derefable type: {}",
                            base_cmt.ty.repr(self.tcx())));
            }
        }
    }

    fn cat_deref_common<N:ast_node>(&mut self,
                                    node: &N,
                                    base_cmt: cmt,
                                    deref_cnt: uint,
                                    deref_ty: ty::t)
                                    -> cmt {
        let (m, cat) = match deref_kind(self.tcx(), base_cmt.ty) {
            deref_ptr(ptr) => {
                // for unique ptrs, we inherit mutability from the
                // owning reference.
                (MutabilityCategory::from_pointer_kind(base_cmt.mutbl, ptr),
                 cat_deref(base_cmt, deref_cnt, ptr))
            }
            deref_interior(interior) => {
                (base_cmt.mutbl.inherit(), cat_interior(base_cmt, interior))
            }
        };
        @cmt_ {
            id: node.id(),
            span: node.span(),
            cat: cat,
            mutbl: m,
            ty: deref_ty
        }
    }

    pub fn cat_index<N:ast_node>(&mut self,
                                 elt: &N,
                                 base_cmt: cmt,
                                 derefs: uint)
                                 -> cmt {
        //! Creates a cmt for an indexing operation (`[]`); this
        //! indexing operation may occurs as part of an
        //! AutoBorrowVec, which when converting a `~[]` to an `&[]`
        //! effectively takes the address of the 0th element.
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
        //! In the event that a deref is needed, the "deref count"
        //! is taken from the parameter `derefs`. See the comment
        //! on the def'n of `root_map_key` in borrowck/mod.rs
        //! for more details about deref counts; the summary is
        //! that `derefs` should be 0 for an explicit indexing
        //! operation and N+1 for an indexing that is part of
        //! an auto-adjustment, where N is the number of autoderefs
        //! in that adjustment.
        //!
        //! # Parameters
        //! - `elt`: the AST node being indexed
        //! - `base_cmt`: the cmt of `elt`
        //! - `derefs`: the deref number to be used for
        //!   the implicit index deref, if any (see above)

        let element_ty = match ty::index(base_cmt.ty) {
          Some(ty) => ty,
          None => {
            self.tcx().sess.span_bug(
                elt.span(),
                format!("Explicit index of non-index type `{}`",
                     base_cmt.ty.repr(self.tcx())));
          }
        };

        return match deref_kind(self.tcx(), base_cmt.ty) {
          deref_ptr(ptr) => {
            // for unique ptrs, we inherit mutability from the
            // owning reference.
            let m = MutabilityCategory::from_pointer_kind(base_cmt.mutbl, ptr);

            // the deref is explicit in the resulting cmt
            let deref_cmt = @cmt_ {
                id:elt.id(),
                span:elt.span(),
                cat:cat_deref(base_cmt, derefs, ptr),
                mutbl:m,
                ty:element_ty
            };

            interior(elt, deref_cmt, base_cmt.ty, m.inherit(), element_ty)
          }

          deref_interior(_) => {
            // fixed-length vectors have no deref
            let m = base_cmt.mutbl.inherit();
            interior(elt, base_cmt, base_cmt.ty, m, element_ty)
          }
        };

        fn interior<N: ast_node>(elt: &N,
                                 of_cmt: cmt,
                                 vec_ty: ty::t,
                                 mutbl: MutabilityCategory,
                                 element_ty: ty::t) -> cmt
        {
            @cmt_ {
                id:elt.id(),
                span:elt.span(),
                cat:cat_interior(of_cmt, InteriorElement(element_kind(vec_ty))),
                mutbl:mutbl,
                ty:element_ty
            }
        }
    }

    pub fn cat_slice_pattern(&mut self,
                             vec_cmt: cmt,
                             slice_pat: @ast::Pat)
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
        let cmt_slice = self.cat_index(slice_pat, vec_cmt, 0);
        return Ok((cmt_slice, slice_mutbl, slice_r));

        fn vec_slice_info(tcx: &ty::ctxt,
                          pat: @ast::Pat,
                          slice_ty: ty::t)
                          -> (ast::Mutability, ty::Region) {
            /*!
             * In a pattern like [a, b, ..c], normally `c` has slice type,
             * but if you have [a, b, ..ref c], then the type of `ref c`
             * will be `&&[]`, so to extract the slice details we have
             * to recurse through rptrs.
             */

            match ty::get(slice_ty).sty {
                ty::ty_vec(_, ty::VstoreSlice(slice_r, mutbl)) => {
                    (mutbl, slice_r)
                }

                ty::ty_rptr(_, ref mt) => {
                    vec_slice_info(tcx, pat, mt.ty)
                }

                _ => {
                    tcx.sess.span_bug(
                        pat.span,
                        format!("Type of slice pattern is not a slice"));
                }
            }
        }
    }

    pub fn cat_imm_interior<N:ast_node>(&mut self,
                                        node: &N,
                                        base_cmt: cmt,
                                        interior_ty: ty::t,
                                        interior: InteriorKind)
                                        -> cmt {
        @cmt_ {
            id: node.id(),
            span: node.span(),
            cat: cat_interior(base_cmt, interior),
            mutbl: base_cmt.mutbl.inherit(),
            ty: interior_ty
        }
    }

    pub fn cat_downcast<N:ast_node>(&mut self,
                                    node: &N,
                                    base_cmt: cmt,
                                    downcast_ty: ty::t)
                                    -> cmt {
        @cmt_ {
            id: node.id(),
            span: node.span(),
            cat: cat_downcast(base_cmt),
            mutbl: base_cmt.mutbl.inherit(),
            ty: downcast_ty
        }
    }

    pub fn cat_pattern(&mut self,
                       cmt: cmt,
                       pat: @ast::Pat,
                       op: |&mut MemCategorizationContext<TYPER>,
                            cmt,
                            @ast::Pat|)
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
               pat.id, pprust::pat_to_str(pat),
               cmt.repr(self.tcx()));

        op(self, cmt, pat);

        match pat.node {
          ast::PatWild | ast::PatWildMulti => {
            // _
          }

          ast::PatEnum(_, None) => {
            // variant(..)
          }
          ast::PatEnum(_, Some(ref subpats)) => {
            match self.tcx().def_map.borrow().find(&pat.id) {
                Some(&ast::DefVariant(enum_did, _, _)) => {
                    // variant(x, y, z)

                    let downcast_cmt = {
                        if ty::enum_is_univariant(self.tcx(), enum_did) {
                            cmt // univariant, no downcast needed
                        } else {
                            self.cat_downcast(pat, cmt, cmt.ty)
                        }
                    };

                    for (i, &subpat) in subpats.iter().enumerate() {
                        let subpat_ty = if_ok!(self.pat_ty(subpat)); // see (*2)

                        let subcmt =
                            self.cat_imm_interior(
                                pat, downcast_cmt, subpat_ty,
                                InteriorField(PositionalField(i)));

                        if_ok!(self.cat_pattern(subcmt, subpat, |x,y,z| op(x,y,z)));
                    }
                }
                Some(&ast::DefFn(..)) |
                Some(&ast::DefStruct(..)) => {
                    for (i, &subpat) in subpats.iter().enumerate() {
                        let subpat_ty = if_ok!(self.pat_ty(subpat)); // see (*2)
                        let cmt_field =
                            self.cat_imm_interior(
                                pat, cmt, subpat_ty,
                                InteriorField(PositionalField(i)));
                        if_ok!(self.cat_pattern(cmt_field, subpat, |x,y,z| op(x,y,z)));
                    }
                }
                Some(&ast::DefStatic(..)) => {
                    for &subpat in subpats.iter() {
                        if_ok!(self.cat_pattern(cmt, subpat, |x,y,z| op(x,y,z)));
                    }
                }
                _ => {
                    self.tcx().sess.span_bug(
                        pat.span,
                        "enum pattern didn't resolve to enum or struct");
                }
            }
          }

          ast::PatIdent(_, _, Some(subpat)) => {
              if_ok!(self.cat_pattern(cmt, subpat, op));
          }

          ast::PatIdent(_, _, None) => {
              // nullary variant or identifier: ignore
          }

          ast::PatStruct(_, ref field_pats, _) => {
            // {f1: p1, ..., fN: pN}
            for fp in field_pats.iter() {
                let field_ty = if_ok!(self.pat_ty(fp.pat)); // see (*2)
                let cmt_field = self.cat_field(pat, cmt, fp.ident, field_ty);
                if_ok!(self.cat_pattern(cmt_field, fp.pat, |x,y,z| op(x,y,z)));
            }
          }

          ast::PatTup(ref subpats) => {
            // (p1, ..., pN)
            for (i, &subpat) in subpats.iter().enumerate() {
                let subpat_ty = if_ok!(self.pat_ty(subpat)); // see (*2)
                let subcmt =
                    self.cat_imm_interior(
                        pat, cmt, subpat_ty,
                        InteriorField(PositionalField(i)));
                if_ok!(self.cat_pattern(subcmt, subpat, |x,y,z| op(x,y,z)));
            }
          }

          ast::PatUniq(subpat) | ast::PatRegion(subpat) => {
            // @p1, ~p1
            let subcmt = self.cat_deref(pat, cmt, 0);
            if_ok!(self.cat_pattern(subcmt, subpat, op));
          }

          ast::PatVec(ref before, slice, ref after) => {
              let elt_cmt = self.cat_index(pat, cmt, 0);
              for &before_pat in before.iter() {
                  if_ok!(self.cat_pattern(elt_cmt, before_pat, |x,y,z| op(x,y,z)));
              }
              for &slice_pat in slice.iter() {
                  let slice_ty = if_ok!(self.pat_ty(slice_pat));
                  let slice_cmt = self.cat_rvalue_node(pat.id(), pat.span(), slice_ty);
                  if_ok!(self.cat_pattern(slice_cmt, slice_pat, |x,y,z| op(x,y,z)));
              }
              for &after_pat in after.iter() {
                  if_ok!(self.cat_pattern(elt_cmt, after_pat, |x,y,z| op(x,y,z)));
              }
          }

          ast::PatLit(_) | ast::PatRange(_, _) => {
              /*always ok*/
          }
        }

        Ok(())
    }

    pub fn cmt_to_str(&self, cmt: cmt) -> ~str {
        match cmt.cat {
          cat_static_item => {
              "static item".to_owned()
          }
          cat_copied_upvar(_) => {
              "captured outer variable in a proc".to_owned()
          }
          cat_rvalue(..) => {
              "non-lvalue".to_owned()
          }
          cat_local(_) => {
              "local variable".to_owned()
          }
          cat_arg(..) => {
              "argument".to_owned()
          }
          cat_deref(base, _, pk) => {
              match base.cat {
                  cat_upvar(..) => {
                      format!("captured outer variable")
                  }
                  _ => {
                      format!("dereference of `{}`-pointer", ptr_sigil(pk))
                  }
              }
          }
          cat_interior(_, InteriorField(NamedField(_))) => {
              "field".to_owned()
          }
          cat_interior(_, InteriorField(PositionalField(_))) => {
              "anonymous field".to_owned()
          }
          cat_interior(_, InteriorElement(VecElement)) => {
              "vec content".to_owned()
          }
          cat_interior(_, InteriorElement(StrElement)) => {
              "str content".to_owned()
          }
          cat_interior(_, InteriorElement(OtherElement)) => {
              "indexed content".to_owned()
          }
          cat_upvar(..) => {
              "captured outer variable".to_owned()
          }
          cat_discr(cmt, _) => {
            self.cmt_to_str(cmt)
          }
          cat_downcast(cmt) => {
            self.cmt_to_str(cmt)
          }
        }
    }
}

pub enum InteriorSafety {
    InteriorUnsafe,
    InteriorSafe
}

pub enum AliasableReason {
    AliasableManaged,
    AliasableBorrowed,
    AliasableOther,
    AliasableStatic(InteriorSafety),
    AliasableStaticMut(InteriorSafety),
}

impl cmt_ {
    pub fn guarantor(self) -> cmt {
        //! Returns `self` after stripping away any owned pointer derefs or
        //! interior content. The return value is basically the `cmt` which
        //! determines how long the value in `self` remains live.

        match self.cat {
            cat_rvalue(..) |
            cat_static_item |
            cat_copied_upvar(..) |
            cat_local(..) |
            cat_arg(..) |
            cat_deref(_, _, UnsafePtr(..)) |
            cat_deref(_, _, GcPtr(..)) |
            cat_deref(_, _, BorrowedPtr(..)) |
            cat_upvar(..) => {
                @self
            }
            cat_downcast(b) |
            cat_discr(b, _) |
            cat_interior(b, _) |
            cat_deref(b, _, OwnedPtr) => {
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
            cat_deref(b, _, BorrowedPtr(ty::MutBorrow, _)) |
            cat_deref(b, _, BorrowedPtr(ty::UniqueImmBorrow, _)) |
            cat_downcast(b) |
            cat_deref(b, _, OwnedPtr) |
            cat_interior(b, _) |
            cat_discr(b, _) => {
                // Aliasability depends on base cmt
                b.freely_aliasable(ctxt)
            }

            cat_copied_upvar(CopiedUpvar {onceness: ast::Once, ..}) |
            cat_rvalue(..) |
            cat_local(..) |
            cat_upvar(..) |
            cat_arg(_) |
            cat_deref(_, _, UnsafePtr(..)) => { // yes, it's aliasable, but...
                None
            }

            cat_copied_upvar(CopiedUpvar {onceness: ast::Many, ..}) => {
                Some(AliasableOther)
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

            cat_deref(_, _, GcPtr) => {
                Some(AliasableManaged)
            }

            cat_deref(_, _, BorrowedPtr(ty::ImmBorrow, _)) => {
                Some(AliasableBorrowed)
            }
        }
    }
}

impl Repr for cmt_ {
    fn repr(&self, tcx: &ty::ctxt) -> ~str {
        format!("\\{{} id:{} m:{:?} ty:{}\\}",
             self.cat.repr(tcx),
             self.id,
             self.mutbl,
             self.ty.repr(tcx))
    }
}

impl Repr for categorization {
    fn repr(&self, tcx: &ty::ctxt) -> ~str {
        match *self {
            cat_static_item |
            cat_rvalue(..) |
            cat_copied_upvar(..) |
            cat_local(..) |
            cat_upvar(..) |
            cat_arg(..) => {
                format!("{:?}", *self)
            }
            cat_deref(cmt, derefs, ptr) => {
                format!("{}-{}{}->",
                        cmt.cat.repr(tcx),
                        ptr_sigil(ptr),
                        derefs)
            }
            cat_interior(cmt, interior) => {
                format!("{}.{}",
                     cmt.cat.repr(tcx),
                     interior.repr(tcx))
            }
            cat_downcast(cmt) => {
                format!("{}->(enum)", cmt.cat.repr(tcx))
            }
            cat_discr(cmt, _) => {
                cmt.cat.repr(tcx)
            }
        }
    }
}

pub fn ptr_sigil(ptr: PointerKind) -> &'static str {
    match ptr {
        OwnedPtr => "~",
        GcPtr => "@",
        BorrowedPtr(ty::ImmBorrow, _) => "&",
        BorrowedPtr(ty::MutBorrow, _) => "&mut",
        BorrowedPtr(ty::UniqueImmBorrow, _) => "&unique",
        UnsafePtr(_) => "*"
    }
}

impl Repr for InteriorKind {
    fn repr(&self, _tcx: &ty::ctxt) -> ~str {
        match *self {
            InteriorField(NamedField(fld)) => {
                token::get_name(fld).get().to_str()
            }
            InteriorField(PositionalField(i)) => format!("\\#{:?}", i),
            InteriorElement(_) => "[]".to_owned(),
        }
    }
}

fn element_kind(t: ty::t) -> ElementKind {
    match ty::get(t).sty {
        ty::ty_vec(..) => VecElement,
        ty::ty_str(..) => StrElement,
        _ => OtherElement
    }
}
