// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
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
 *       | x         // address of a local variable, arg, or upvar
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
 */


use middle::ty;
use middle::typeck;
use util::ppaux::{ty_to_str, region_ptr_to_str, Repr};
use util::common::indenter;

use syntax::ast::{MutImmutable, MutMutable};
use syntax::ast;
use syntax::codemap::Span;
use syntax::print::pprust;
use syntax::parse::token;

#[deriving(Eq)]
pub enum categorization {
    cat_rvalue(ast::NodeId),           // temporary val, argument is its scope
    cat_static_item,
    cat_copied_upvar(CopiedUpvar),     // upvar copied into @fn or ~fn env
    cat_stack_upvar(cmt),              // by ref upvar from &fn
    cat_local(ast::NodeId),            // local variable
    cat_arg(ast::NodeId),              // formal argument
    cat_deref(cmt, uint, PointerKind), // deref of a ptr
    cat_interior(cmt, InteriorKind),   // something interior: field, tuple, etc
    cat_downcast(cmt),                 // selects a particular enum variant (*)
    cat_discr(cmt, ast::NodeId),       // match discriminant (see preserve())
    cat_self(ast::NodeId),             // explicit `self`

    // (*) downcast is only required if the enum has more than one variant
}

#[deriving(Eq)]
pub struct CopiedUpvar {
    upvar_id: ast::NodeId,
    onceness: ast::Onceness,
}

// different kinds of pointers:
#[deriving(Eq, IterBytes)]
pub enum PointerKind {
    uniq_ptr,
    gc_ptr(ast::Mutability),
    region_ptr(ast::Mutability, ty::Region),
    unsafe_ptr(ast::Mutability)
}

// We use the term "interior" to mean "something reachable from the
// base without a pointer dereference", e.g. a field
#[deriving(Eq, IterBytes)]
pub enum InteriorKind {
    InteriorField(FieldName),
    InteriorElement(ElementKind),
}

#[deriving(Eq, IterBytes)]
pub enum FieldName {
    NamedField(ast::Name),
    PositionalField(uint)
}

#[deriving(Eq, IterBytes)]
pub enum ElementKind {
    VecElement,
    StrElement,
    OtherElement,
}

#[deriving(Eq, IterBytes)]
pub enum MutabilityCategory {
    McImmutable, // Immutable.
    McDeclared,  // Directly declared as mutable.
    McInherited  // Inherited from the fact that owner is mutable.
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
    id: ast::NodeId,          // id of expr/pat producing this value
    span: Span,                // span of same expr/pat
    cat: categorization,       // categorization of expr
    mutbl: MutabilityCategory, // mutability of expr as lvalue
    ty: ty::t                  // type of the expr (*see WARNING above*)
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
        ty::ty_trait(_, _, ty::UniqTraitStore, _, _) |
        ty::ty_evec(_, ty::vstore_uniq) |
        ty::ty_estr(ty::vstore_uniq) |
        ty::ty_closure(ty::ClosureTy {sigil: ast::OwnedSigil, _}) => {
            Some(deref_ptr(uniq_ptr))
        }

        ty::ty_rptr(r, mt) |
        ty::ty_evec(mt, ty::vstore_slice(r)) => {
            Some(deref_ptr(region_ptr(mt.mutbl, r)))
        }

        ty::ty_trait(_, _, ty::RegionTraitStore(r), m, _) => {
            Some(deref_ptr(region_ptr(m, r)))
        }

        ty::ty_estr(ty::vstore_slice(r)) |
        ty::ty_closure(ty::ClosureTy {sigil: ast::BorrowedSigil,
                                      region: r, _}) => {
            Some(deref_ptr(region_ptr(ast::MutImmutable, r)))
        }

        ty::ty_box(ref mt) |
        ty::ty_evec(ref mt, ty::vstore_box) => {
            Some(deref_ptr(gc_ptr(mt.mutbl)))
        }

        ty::ty_trait(_, _, ty::BoxTraitStore, m, _) => {
            Some(deref_ptr(gc_ptr(m)))
        }

        ty::ty_estr(ty::vstore_box) => {
            Some(deref_ptr(gc_ptr(ast::MutImmutable)))
        }

        ty::ty_ptr(ref mt) => {
            Some(deref_ptr(unsafe_ptr(mt.mutbl)))
        }

        ty::ty_enum(*) |
        ty::ty_struct(*) => { // newtype
            Some(deref_interior(InteriorField(PositionalField(0))))
        }

        ty::ty_evec(_, ty::vstore_fixed(_)) |
        ty::ty_estr(ty::vstore_fixed(_)) => {
            Some(deref_interior(InteriorElement(element_kind(t))))
        }

        _ => None
    }
}

pub fn deref_kind(tcx: ty::ctxt, t: ty::t) -> deref_kind {
    match opt_deref_kind(t) {
      Some(k) => k,
      None => {
        tcx.sess.bug(
            format!("deref_cat() invoked on non-derefable type {}",
                 ty_to_str(tcx, t)));
      }
    }
}

pub fn cat_expr(tcx: ty::ctxt,
                method_map: typeck::method_map,
                expr: @ast::Expr)
             -> cmt {
    let mcx = &mem_categorization_ctxt {
        tcx: tcx, method_map: method_map
    };
    return mcx.cat_expr(expr);
}

pub fn cat_expr_unadjusted(tcx: ty::ctxt,
                           method_map: typeck::method_map,
                           expr: @ast::Expr)
                        -> cmt {
    let mcx = &mem_categorization_ctxt {
        tcx: tcx, method_map: method_map
    };
    return mcx.cat_expr_unadjusted(expr);
}

pub fn cat_expr_autoderefd(
    tcx: ty::ctxt,
    method_map: typeck::method_map,
    expr: @ast::Expr,
    autoderefs: uint) -> cmt
{
    let mcx = &mem_categorization_ctxt {
        tcx: tcx, method_map: method_map
    };
    return mcx.cat_expr_autoderefd(expr, autoderefs);
}

pub fn cat_def(
    tcx: ty::ctxt,
    method_map: typeck::method_map,
    expr_id: ast::NodeId,
    expr_span: Span,
    expr_ty: ty::t,
    def: ast::Def) -> cmt {

    let mcx = &mem_categorization_ctxt {
        tcx: tcx, method_map: method_map
    };
    return mcx.cat_def(expr_id, expr_span, expr_ty, def);
}

pub trait ast_node {
    fn id(&self) -> ast::NodeId;
    fn span(&self) -> Span;
}

impl ast_node for @ast::Expr {
    fn id(&self) -> ast::NodeId { self.id }
    fn span(&self) -> Span { self.span }
}

impl ast_node for @ast::Pat {
    fn id(&self) -> ast::NodeId { self.id }
    fn span(&self) -> Span { self.span }
}

pub struct mem_categorization_ctxt {
    tcx: ty::ctxt,
    method_map: typeck::method_map,
}

impl ToStr for MutabilityCategory {
    fn to_str(&self) -> ~str {
        format!("{:?}", *self)
    }
}

impl MutabilityCategory {
    pub fn from_mutbl(m: ast::Mutability) -> MutabilityCategory {
        match m {
            MutImmutable => McImmutable,
            MutMutable => McDeclared
        }
    }

    pub fn inherit(&self) -> MutabilityCategory {
        match *self {
            McImmutable => McImmutable,
            McDeclared => McInherited,
            McInherited => McInherited
        }
    }

    pub fn is_mutable(&self) -> bool {
        match *self {
            McImmutable => false,
            McDeclared | McInherited => true
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

impl mem_categorization_ctxt {
    pub fn expr_ty(&self, expr: @ast::Expr) -> ty::t {
        ty::expr_ty(self.tcx, expr)
    }

    pub fn pat_ty(&self, pat: @ast::Pat) -> ty::t {
        ty::node_id_to_type(self.tcx, pat.id)
    }

    pub fn cat_expr(&self, expr: @ast::Expr) -> cmt {
        match self.tcx.adjustments.find(&expr.id) {
            None => {
                // No adjustments.
                self.cat_expr_unadjusted(expr)
            }

            Some(&@ty::AutoAddEnv(*)) => {
                // Convert a bare fn to a closure by adding NULL env.
                // Result is an rvalue.
                let expr_ty = ty::expr_ty_adjusted(self.tcx, expr);
                self.cat_rvalue_node(expr, expr_ty)
            }

            Some(
                &@ty::AutoDerefRef(
                    ty::AutoDerefRef {
                        autoref: Some(_), _})) => {
                // Equivalent to &*expr or something similar.
                // Result is an rvalue.
                let expr_ty = ty::expr_ty_adjusted(self.tcx, expr);
                self.cat_rvalue_node(expr, expr_ty)
            }

            Some(
                &@ty::AutoDerefRef(
                    ty::AutoDerefRef {
                        autoref: None, autoderefs: autoderefs})) => {
                // Equivalent to *expr or something similar.
                self.cat_expr_autoderefd(expr, autoderefs)
            }
        }
    }

    pub fn cat_expr_autoderefd(&self, expr: @ast::Expr, autoderefs: uint)
                               -> cmt {
        let mut cmt = self.cat_expr_unadjusted(expr);
        for deref in range(1u, autoderefs + 1) {
            cmt = self.cat_deref(expr, cmt, deref);
        }
        return cmt;
    }

    pub fn cat_expr_unadjusted(&self, expr: @ast::Expr) -> cmt {
        debug!("cat_expr: id={} expr={}",
               expr.id, pprust::expr_to_str(expr, self.tcx.sess.intr()));

        let expr_ty = self.expr_ty(expr);
        match expr.node {
          ast::ExprUnary(_, ast::UnDeref, e_base) => {
            if self.method_map.contains_key(&expr.id) {
                return self.cat_rvalue_node(expr, expr_ty);
            }

            let base_cmt = self.cat_expr(e_base);
            self.cat_deref(expr, base_cmt, 0)
          }

          ast::ExprField(base, f_name, _) => {
            // Method calls are now a special syntactic form,
            // so `a.b` should always be a field.
            assert!(!self.method_map.contains_key(&expr.id));

            let base_cmt = self.cat_expr(base);
            self.cat_field(expr, base_cmt, f_name, self.expr_ty(expr))
          }

          ast::ExprIndex(_, base, _) => {
            if self.method_map.contains_key(&expr.id) {
                return self.cat_rvalue_node(expr, expr_ty);
            }

            let base_cmt = self.cat_expr(base);
            self.cat_index(expr, base_cmt, 0)
          }

          ast::ExprPath(_) | ast::ExprSelf => {
            let def = self.tcx.def_map.get_copy(&expr.id);
            self.cat_def(expr.id, expr.span, expr_ty, def)
          }

          ast::ExprParen(e) => self.cat_expr_unadjusted(e),

          ast::ExprAddrOf(*) | ast::ExprCall(*) |
          ast::ExprAssign(*) | ast::ExprAssignOp(*) |
          ast::ExprFnBlock(*) | ast::ExprRet(*) |
          ast::ExprDoBody(*) | ast::ExprUnary(*) |
          ast::ExprMethodCall(*) | ast::ExprCast(*) | ast::ExprVstore(*) |
          ast::ExprVec(*) | ast::ExprTup(*) | ast::ExprIf(*) |
          ast::ExprLogLevel | ast::ExprBinary(*) | ast::ExprWhile(*) |
          ast::ExprBlock(*) | ast::ExprLoop(*) | ast::ExprMatch(*) |
          ast::ExprLit(*) | ast::ExprBreak(*) | ast::ExprMac(*) |
          ast::ExprAgain(*) | ast::ExprStruct(*) | ast::ExprRepeat(*) |
          ast::ExprInlineAsm(*) => {
            return self.cat_rvalue_node(expr, expr_ty);
          }

          ast::ExprForLoop(*) => fail!("non-desugared expr_for_loop")
        }
    }

    pub fn cat_def(&self,
                   id: ast::NodeId,
                   span: Span,
                   expr_ty: ty::t,
                   def: ast::Def)
                   -> cmt {
        match def {
          ast::DefFn(*) | ast::DefStaticMethod(*) | ast::DefMod(_) |
          ast::DefForeignMod(_) | ast::DefStatic(_, false) |
          ast::DefUse(_) | ast::DefVariant(*) |
          ast::DefTrait(_) | ast::DefTy(_) | ast::DefPrimTy(_) |
          ast::DefTyParam(*) | ast::DefStruct(*) |
          ast::DefTyParamBinder(*) | ast::DefRegion(_) |
          ast::DefLabel(_) | ast::DefSelfTy(*) | ast::DefMethod(*) => {
              @cmt_ {
                  id:id,
                  span:span,
                  cat:cat_static_item,
                  mutbl: McImmutable,
                  ty:expr_ty
              }
          }

          ast::DefStatic(_, true) => {
              @cmt_ {
                  id:id,
                  span:span,
                  cat:cat_static_item,
                  mutbl: McDeclared,
                  ty:expr_ty
              }
          }

          ast::DefArg(vid, binding_mode) => {
            // Idea: make this could be rewritten to model by-ref
            // stuff as `&const` and `&mut`?

            // m: mutability of the argument
            let m = match binding_mode {
                ast::BindByValue(ast::MutMutable) => McDeclared,
                _ => McImmutable
            };
            @cmt_ {
                id: id,
                span: span,
                cat: cat_arg(vid),
                mutbl: m,
                ty:expr_ty
            }
          }

          ast::DefSelf(self_id, mutbl) => {
            @cmt_ {
                id:id,
                span:span,
                cat:cat_self(self_id),
                mutbl: if mutbl { McDeclared } else { McImmutable },
                ty:expr_ty
            }
          }

          ast::DefUpvar(upvar_id, inner, fn_node_id, _) => {
              let ty = ty::node_id_to_type(self.tcx, fn_node_id);
              match ty::get(ty).sty {
                  ty::ty_closure(ref closure_ty) => {
                      // Decide whether to use implicit reference or by copy/move
                      // capture for the upvar. This, combined with the onceness,
                      // determines whether the closure can move out of it.
                      let var_is_refd = match (closure_ty.sigil, closure_ty.onceness) {
                          // Many-shot stack closures can never move out.
                          (ast::BorrowedSigil, ast::Many) => true,
                          // 1-shot stack closures can move out.
                          (ast::BorrowedSigil, ast::Once) => false,
                          // Heap closures always capture by copy/move, and can
                          // move out if they are once.
                          (ast::OwnedSigil, _) |
                          (ast::ManagedSigil, _) => false,

                      };
                      if var_is_refd {
                          let upvar_cmt =
                              self.cat_def(id, span, expr_ty, *inner);
                          @cmt_ {
                              id:id,
                              span:span,
                              cat:cat_stack_upvar(upvar_cmt),
                              mutbl:upvar_cmt.mutbl.inherit(),
                              ty:upvar_cmt.ty
                          }
                      } else {
                          // FIXME #2152 allow mutation of moved upvars
                          @cmt_ {
                              id:id,
                              span:span,
                              cat:cat_copied_upvar(CopiedUpvar {
                                  upvar_id: upvar_id,
                                  onceness: closure_ty.onceness}),
                              mutbl:McImmutable,
                              ty:expr_ty
                          }
                      }
                  }
                  _ => {
                      self.tcx.sess.span_bug(
                          span,
                          format!("Upvar of non-closure {:?} - {}",
                               fn_node_id, ty.repr(self.tcx)));
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

            @cmt_ {
                id: id,
                span: span,
                cat: cat_local(vid),
                mutbl: m,
                ty: expr_ty
            }
          }
        }
    }

    pub fn cat_rvalue_node<N:ast_node>(&self,
                                       node: N,
                                       expr_ty: ty::t) -> cmt {
        self.cat_rvalue(node.id(),
                        node.span(),
                        self.tcx.region_maps.cleanup_scope(node.id()),
                        expr_ty)
    }

    pub fn cat_rvalue(&self,
                      cmt_id: ast::NodeId,
                      span: Span,
                      cleanup_scope_id: ast::NodeId,
                      expr_ty: ty::t) -> cmt {
        @cmt_ {
            id:cmt_id,
            span:span,
            cat:cat_rvalue(cleanup_scope_id),
            mutbl:McDeclared,
            ty:expr_ty
        }
    }

    /// inherited mutability: used in cases where the mutability of a
    /// component is inherited from the base it is a part of. For
    /// example, a record field is mutable if it is declared mutable
    /// or if the container is mutable.
    pub fn inherited_mutability(&self,
                                base_m: MutabilityCategory,
                                interior_m: ast::Mutability)
                                -> MutabilityCategory {
        match interior_m {
            MutImmutable => base_m.inherit(),
            MutMutable => McDeclared
        }
    }

    pub fn cat_field<N:ast_node>(&self,
                                 node: N,
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

    pub fn cat_deref_fn_or_obj<N:ast_node>(&self,
                                           node: N,
                                           base_cmt: cmt,
                                           deref_cnt: uint)
                                           -> cmt {
        // Bit of a hack: the "dereference" of a function pointer like
        // `@fn()` is a mere logical concept. We interpret it as
        // dereferencing the environment pointer; of course, we don't
        // know what type lies at the other end, so we just call it
        // `()` (the empty tuple).

        let opaque_ty = ty::mk_tup(self.tcx, ~[]);
        return self.cat_deref_common(node, base_cmt, deref_cnt, opaque_ty);
    }

    pub fn cat_deref<N:ast_node>(&self,
                                 node: N,
                                 base_cmt: cmt,
                                 deref_cnt: uint)
                                 -> cmt {
        let mt = match ty::deref(self.tcx, base_cmt.ty, true) {
            Some(mt) => mt,
            None => {
                self.tcx.sess.span_bug(
                    node.span(),
                    format!("Explicit deref of non-derefable type: {}",
                         ty_to_str(self.tcx, base_cmt.ty)));
            }
        };

        return self.cat_deref_common(node, base_cmt, deref_cnt, mt.ty);
    }

    pub fn cat_deref_common<N:ast_node>(&self,
                                        node: N,
                                        base_cmt: cmt,
                                        deref_cnt: uint,
                                        deref_ty: ty::t)
                                        -> cmt {
        match deref_kind(self.tcx, base_cmt.ty) {
            deref_ptr(ptr) => {
                // for unique ptrs, we inherit mutability from the
                // owning reference.
                let m = match ptr {
                    uniq_ptr => {
                        base_cmt.mutbl.inherit()
                    }
                    gc_ptr(m) | region_ptr(m, _) | unsafe_ptr(m) => {
                        MutabilityCategory::from_mutbl(m)
                    }
                };

                @cmt_ {
                    id:node.id(),
                    span:node.span(),
                    cat:cat_deref(base_cmt, deref_cnt, ptr),
                    mutbl:m,
                    ty:deref_ty
                }
            }

            deref_interior(interior) => {
                let m = base_cmt.mutbl.inherit();
                @cmt_ {
                    id:node.id(),
                    span:node.span(),
                    cat:cat_interior(base_cmt, interior),
                    mutbl:m,
                    ty:deref_ty
                }
            }
        }
    }

    pub fn cat_index<N:ast_node>(&self,
                                 elt: N,
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
          Some(ref mt) => mt.ty,
          None => {
            self.tcx.sess.span_bug(
                elt.span(),
                format!("Explicit index of non-index type `{}`",
                     ty_to_str(self.tcx, base_cmt.ty)));
          }
        };

        return match deref_kind(self.tcx, base_cmt.ty) {
          deref_ptr(ptr) => {
            // for unique ptrs, we inherit mutability from the
            // owning reference.
            let m = match ptr {
              uniq_ptr => {
                base_cmt.mutbl.inherit()
              }
              gc_ptr(m) | region_ptr(m, _) | unsafe_ptr(m) => {
                MutabilityCategory::from_mutbl(m)
              }
            };

            // the deref is explicit in the resulting cmt
            let deref_cmt = @cmt_ {
                id:elt.id(),
                span:elt.span(),
                cat:cat_deref(base_cmt, derefs, ptr),
                mutbl:m,
                ty:element_ty
            };

            interior(elt, deref_cmt, base_cmt.ty, m, element_ty)
          }

          deref_interior(_) => {
            // fixed-length vectors have no deref
            let m = base_cmt.mutbl.inherit();
            interior(elt, base_cmt, base_cmt.ty, m, element_ty)
          }
        };

        fn interior<N: ast_node>(elt: N,
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

    pub fn cat_imm_interior<N:ast_node>(&self,
                                        node: N,
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

    pub fn cat_downcast<N:ast_node>(&self,
                                    node: N,
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

    pub fn cat_pattern(&self,
                       cmt: cmt,
                       pat: @ast::Pat,
                       op: &fn(cmt, @ast::Pat)) {
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
        // (*) There is subtlety concerning the correspondence between
        // pattern ids and types as compared to *expression* ids and
        // types. This is explained briefly. on the definition of the
        // type `cmt`, so go off and read what it says there, then
        // come back and I'll dive into a bit more detail here. :) OK,
        // back?
        //
        // In general, the id of the cmt should be the node that
        // "produces" the value---patterns aren't executable code
        // exactly, but I consider them to "execute" when they match a
        // value. So if you have something like:
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

        let tcx = self.tcx;
        debug!("cat_pattern: id={} pat={} cmt={}",
               pat.id, pprust::pat_to_str(pat, tcx.sess.intr()),
               cmt.repr(tcx));
        let _i = indenter();

        op(cmt, pat);

        match pat.node {
          ast::PatWild => {
            // _
          }

          ast::PatEnum(_, None) => {
            // variant(*)
          }
          ast::PatEnum(_, Some(ref subpats)) => {
            match self.tcx.def_map.find(&pat.id) {
                Some(&ast::DefVariant(enum_did, _, _)) => {
                    // variant(x, y, z)

                    let downcast_cmt = {
                        if ty::enum_is_univariant(tcx, enum_did) {
                            cmt // univariant, no downcast needed
                        } else {
                            self.cat_downcast(pat, cmt, cmt.ty)
                        }
                    };

                    for (i, &subpat) in subpats.iter().enumerate() {
                        let subpat_ty = self.pat_ty(subpat); // see (*)

                        let subcmt =
                            self.cat_imm_interior(
                                pat, downcast_cmt, subpat_ty,
                                InteriorField(PositionalField(i)));

                        self.cat_pattern(subcmt, subpat, |x,y| op(x,y));
                    }
                }
                Some(&ast::DefFn(*)) |
                Some(&ast::DefStruct(*)) => {
                    for (i, &subpat) in subpats.iter().enumerate() {
                        let subpat_ty = self.pat_ty(subpat); // see (*)
                        let cmt_field =
                            self.cat_imm_interior(
                                pat, cmt, subpat_ty,
                                InteriorField(PositionalField(i)));
                        self.cat_pattern(cmt_field, subpat, |x,y| op(x,y));
                    }
                }
                Some(&ast::DefStatic(*)) => {
                    for &subpat in subpats.iter() {
                        self.cat_pattern(cmt, subpat, |x,y| op(x,y));
                    }
                }
                _ => {
                    self.tcx.sess.span_bug(
                        pat.span,
                        "enum pattern didn't resolve to enum or struct");
                }
            }
          }

          ast::PatIdent(_, _, Some(subpat)) => {
              self.cat_pattern(cmt, subpat, op);
          }

          ast::PatIdent(_, _, None) => {
              // nullary variant or identifier: ignore
          }

          ast::PatStruct(_, ref field_pats, _) => {
            // {f1: p1, ..., fN: pN}
            for fp in field_pats.iter() {
                let field_ty = self.pat_ty(fp.pat); // see (*)
                let cmt_field = self.cat_field(pat, cmt, fp.ident, field_ty);
                self.cat_pattern(cmt_field, fp.pat, |x,y| op(x,y));
            }
          }

          ast::PatTup(ref subpats) => {
            // (p1, ..., pN)
            for (i, &subpat) in subpats.iter().enumerate() {
                let subpat_ty = self.pat_ty(subpat); // see (*)
                let subcmt =
                    self.cat_imm_interior(
                        pat, cmt, subpat_ty,
                        InteriorField(PositionalField(i)));
                self.cat_pattern(subcmt, subpat, |x,y| op(x,y));
            }
          }

          ast::PatBox(subpat) | ast::PatUniq(subpat) |
          ast::PatRegion(subpat) => {
            // @p1, ~p1
            let subcmt = self.cat_deref(pat, cmt, 0);
            self.cat_pattern(subcmt, subpat, op);
          }

          ast::PatVec(ref before, slice, ref after) => {
              let elt_cmt = self.cat_index(pat, cmt, 0);
              for &before_pat in before.iter() {
                  self.cat_pattern(elt_cmt, before_pat, |x,y| op(x,y));
              }
              for &slice_pat in slice.iter() {
                  let slice_ty = self.pat_ty(slice_pat);
                  let slice_cmt = self.cat_rvalue_node(pat, slice_ty);
                  self.cat_pattern(slice_cmt, slice_pat, |x,y| op(x,y));
              }
              for &after_pat in after.iter() {
                  self.cat_pattern(elt_cmt, after_pat, |x,y| op(x,y));
              }
          }

          ast::PatLit(_) | ast::PatRange(_, _) => {
              /*always ok*/
          }
        }
    }

    pub fn mut_to_str(&self, mutbl: ast::Mutability) -> ~str {
        match mutbl {
          MutMutable => ~"mutable",
          MutImmutable => ~"immutable"
        }
    }

    pub fn cmt_to_str(&self, cmt: cmt) -> ~str {
        match cmt.cat {
          cat_static_item => {
              ~"static item"
          }
          cat_copied_upvar(_) => {
              ~"captured outer variable in a heap closure"
          }
          cat_rvalue(*) => {
              ~"non-lvalue"
          }
          cat_local(_) => {
              ~"local variable"
          }
          cat_self(_) => {
              ~"self value"
          }
          cat_arg(*) => {
              ~"argument"
          }
          cat_deref(_, _, pk) => {
              format!("dereference of {} pointer", ptr_sigil(pk))
          }
          cat_interior(_, InteriorField(NamedField(_))) => {
              ~"field"
          }
          cat_interior(_, InteriorField(PositionalField(_))) => {
              ~"anonymous field"
          }
          cat_interior(_, InteriorElement(VecElement)) => {
              ~"vec content"
          }
          cat_interior(_, InteriorElement(StrElement)) => {
              ~"str content"
          }
          cat_interior(_, InteriorElement(OtherElement)) => {
              ~"indexed content"
          }
          cat_stack_upvar(_) => {
              ~"captured outer variable"
          }
          cat_discr(cmt, _) => {
            self.cmt_to_str(cmt)
          }
          cat_downcast(cmt) => {
            self.cmt_to_str(cmt)
          }
        }
    }

    pub fn region_to_str(&self, r: ty::Region) -> ~str {
        region_ptr_to_str(self.tcx, r)
    }
}

/// The node_id here is the node of the expression that references the field.
/// This function looks it up in the def map in case the type happens to be
/// an enum to determine which variant is in use.
pub fn field_mutbl(tcx: ty::ctxt,
                   base_ty: ty::t,
                   // FIXME #6993: change type to Name
                   f_name: ast::Ident,
                   node_id: ast::NodeId)
                -> Option<ast::Mutability> {
    // Need to refactor so that struct/enum fields can be treated uniformly.
    match ty::get(base_ty).sty {
      ty::ty_struct(did, _) => {
        let r = ty::lookup_struct_fields(tcx, did);
        for fld in r.iter() {
            if fld.name == f_name.name {
                return Some(ast::MutImmutable);
            }
        }
      }
      ty::ty_enum(*) => {
        match tcx.def_map.get_copy(&node_id) {
          ast::DefVariant(_, variant_id, _) => {
            let r = ty::lookup_struct_fields(tcx, variant_id);
            for fld in r.iter() {
                if fld.name == f_name.name {
                    return Some(ast::MutImmutable);
                }
            }
          }
          _ => {}
        }
      }
      _ => { }
    }

    return None;
}

pub enum AliasableReason {
    AliasableManaged(ast::Mutability),
    AliasableBorrowed(ast::Mutability),
    AliasableOther
}

impl cmt_ {
    pub fn guarantor(@self) -> cmt {
        //! Returns `self` after stripping away any owned pointer derefs or
        //! interior content. The return value is basically the `cmt` which
        //! determines how long the value in `self` remains live.

        match self.cat {
            cat_rvalue(*) |
            cat_static_item |
            cat_copied_upvar(*) |
            cat_local(*) |
            cat_self(*) |
            cat_arg(*) |
            cat_deref(_, _, unsafe_ptr(*)) |
            cat_deref(_, _, gc_ptr(*)) |
            cat_deref(_, _, region_ptr(*)) => {
                self
            }
            cat_downcast(b) |
            cat_stack_upvar(b) |
            cat_discr(b, _) |
            cat_interior(b, _) |
            cat_deref(b, _, uniq_ptr) => {
                b.guarantor()
            }
        }
    }

    pub fn is_freely_aliasable(&self) -> bool {
        self.freely_aliasable().is_some()
    }

    pub fn freely_aliasable(&self) -> Option<AliasableReason> {
        /*!
         * Returns `Some(_)` if this lvalue represents a freely aliasable
         * pointer type.
         */

        // Maybe non-obvious: copied upvars can only be considered
        // non-aliasable in once closures, since any other kind can be
        // aliased and eventually recused.

        match self.cat {
            cat_copied_upvar(CopiedUpvar {onceness: ast::Once, _}) |
            cat_rvalue(*) |
            cat_local(*) |
            cat_arg(_) |
            cat_self(*) |
            cat_deref(_, _, unsafe_ptr(*)) | // of course it is aliasable, but...
            cat_deref(_, _, region_ptr(MutMutable, _)) => {
                None
            }

            cat_copied_upvar(CopiedUpvar {onceness: ast::Many, _}) |
            cat_static_item(*) => {
                Some(AliasableOther)
            }

            cat_deref(_, _, gc_ptr(m)) => {
                Some(AliasableManaged(m))
            }

            cat_deref(_, _, region_ptr(m @ MutImmutable, _)) => {
                Some(AliasableBorrowed(m))
            }

            cat_downcast(*) |
            cat_stack_upvar(*) |
            cat_deref(_, _, uniq_ptr) |
            cat_interior(*) |
            cat_discr(*) => {
                None
            }
        }
    }
}

impl Repr for cmt_ {
    fn repr(&self, tcx: ty::ctxt) -> ~str {
        format!("\\{{} id:{} m:{:?} ty:{}\\}",
             self.cat.repr(tcx),
             self.id,
             self.mutbl,
             self.ty.repr(tcx))
    }
}

impl Repr for categorization {
    fn repr(&self, tcx: ty::ctxt) -> ~str {
        match *self {
            cat_static_item |
            cat_rvalue(*) |
            cat_copied_upvar(*) |
            cat_local(*) |
            cat_self(*) |
            cat_arg(*) => {
                format!("{:?}", *self)
            }
            cat_deref(cmt, derefs, ptr) => {
                format!("{}->({}, {})", cmt.cat.repr(tcx),
                     ptr_sigil(ptr), derefs)
            }
            cat_interior(cmt, interior) => {
                format!("{}.{}",
                     cmt.cat.repr(tcx),
                     interior.repr(tcx))
            }
            cat_downcast(cmt) => {
                format!("{}->(enum)", cmt.cat.repr(tcx))
            }
            cat_stack_upvar(cmt) |
            cat_discr(cmt, _) => {
                cmt.cat.repr(tcx)
            }
        }
    }
}

pub fn ptr_sigil(ptr: PointerKind) -> ~str {
    match ptr {
        uniq_ptr => ~"~",
        gc_ptr(_) => ~"@",
        region_ptr(_, _) => ~"&",
        unsafe_ptr(_) => ~"*"
    }
}

impl Repr for InteriorKind {
    fn repr(&self, _tcx: ty::ctxt) -> ~str {
        match *self {
            InteriorField(NamedField(fld)) => token::interner_get(fld).to_owned(),
            InteriorField(PositionalField(i)) => format!("\\#{:?}", i),
            InteriorElement(_) => ~"[]",
        }
    }
}

fn element_kind(t: ty::t) -> ElementKind {
    match ty::get(t).sty {
        ty::ty_evec(*) => VecElement,
        ty::ty_estr(*) => StrElement,
        _ => OtherElement
    }
}
