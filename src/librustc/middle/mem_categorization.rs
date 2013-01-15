// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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

use core::prelude::*;

use middle::ty;
use middle::typeck;
use util::ppaux::{ty_to_str, region_to_str};
use util::common::indenter;

use core::cmp;
use core::to_bytes;
use core::uint;
use syntax::ast::{m_imm, m_const, m_mutbl};
use syntax::ast;
use syntax::codemap::span;
use syntax::print::pprust;

enum categorization {
    cat_rvalue,                     // result of eval'ing some misc expr
    cat_special(special_kind),      //
    cat_local(ast::node_id),        // local variable
    cat_binding(ast::node_id),      // pattern binding
    cat_arg(ast::node_id),          // formal argument
    cat_stack_upvar(cmt),           // upvar in stack closure
    cat_deref(cmt, uint, ptr_kind), // deref of a ptr
    cat_comp(cmt, comp_kind),       // adjust to locate an internal component
    cat_discr(cmt, ast::node_id),   // match discriminant (see preserve())
}

impl categorization : cmp::Eq {
    pure fn eq(&self, other: &categorization) -> bool {
        match (*self) {
            cat_rvalue => {
                match (*other) {
                    cat_rvalue => true,
                    _ => false
                }
            }
            cat_special(e0a) => {
                match (*other) {
                    cat_special(e0b) => e0a == e0b,
                    _ => false
                }
            }
            cat_local(e0a) => {
                match (*other) {
                    cat_local(e0b) => e0a == e0b,
                    _ => false
                }
            }
            cat_binding(e0a) => {
                match (*other) {
                    cat_binding(e0b) => e0a == e0b,
                    _ => false
                }
            }
            cat_arg(e0a) => {
                match (*other) {
                    cat_arg(e0b) => e0a == e0b,
                    _ => false
                }
            }
            cat_stack_upvar(e0a) => {
                match (*other) {
                    cat_stack_upvar(e0b) => e0a == e0b,
                    _ => false
                }
            }
            cat_deref(e0a, e1a, e2a) => {
                match (*other) {
                    cat_deref(e0b, e1b, e2b) =>
                        e0a == e0b && e1a == e1b && e2a == e2b,
                    _ => false
                }
            }
            cat_comp(e0a, e1a) => {
                match (*other) {
                    cat_comp(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            cat_discr(e0a, e1a) => {
                match (*other) {
                    cat_discr(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
        }
    }
    pure fn ne(&self, other: &categorization) -> bool { !(*self).eq(other) }
}

// different kinds of pointers:
pub enum ptr_kind {
    uniq_ptr,
    gc_ptr,
    region_ptr(ty::Region),
    unsafe_ptr
}

impl ptr_kind : cmp::Eq {
    pure fn eq(&self, other: &ptr_kind) -> bool {
        match (*self) {
            uniq_ptr => {
                match (*other) {
                    uniq_ptr => true,
                    _ => false
                }
            }
            gc_ptr => {
                match (*other) {
                    gc_ptr => true,
                    _ => false
                }
            }
            region_ptr(e0a) => {
                match (*other) {
                    region_ptr(e0b) => e0a == e0b,
                    _ => false
                }
            }
            unsafe_ptr => {
                match (*other) {
                    unsafe_ptr => true,
                    _ => false
                }
            }
        }
    }
    pure fn ne(&self, other: &ptr_kind) -> bool { !(*self).eq(other) }
}

// I am coining the term "components" to mean "pieces of a data
// structure accessible without a dereference":
pub enum comp_kind {
    comp_tuple,                  // elt in a tuple
    comp_anon_field,             // anonymous field (in e.g.
                                 // struct Foo(int, int);
    comp_variant(ast::def_id),   // internals to a variant of given enum
    comp_field(ast::ident,       // name of field
               ast::mutability), // declared mutability of field
    comp_index(ty::t,            // type of vec/str/etc being deref'd
               ast::mutability)  // mutability of vec content
}

impl comp_kind : cmp::Eq {
    pure fn eq(&self, other: &comp_kind) -> bool {
        match (*self) {
            comp_tuple => {
                match (*other) {
                    comp_tuple => true,
                    _ => false
                }
            }
            comp_anon_field => {
                match (*other) {
                    comp_anon_field => true,
                    _ => false
                }
            }
            comp_variant(e0a) => {
                match (*other) {
                    comp_variant(e0b) => e0a == e0b,
                    _ => false
                }
            }
            comp_field(e0a, e1a) => {
                match (*other) {
                    comp_field(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            comp_index(e0a, e1a) => {
                match (*other) {
                    comp_index(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
        }
    }
    pure fn ne(&self, other: &comp_kind) -> bool { !(*self).eq(other) }
}

// different kinds of expressions we might evaluate
enum special_kind {
    sk_method,
    sk_static_item,
    sk_self,
    sk_implicit_self,   // old by-reference `self`
    sk_heap_upvar
}

impl special_kind : cmp::Eq {
    pure fn eq(&self, other: &special_kind) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    pure fn ne(&self, other: &special_kind) -> bool { !(*self).eq(other) }
}

// a complete categorization of a value indicating where it originated
// and how it is located, as well as the mutability of the memory in
// which the value is stored.
type cmt_ = {id: ast::node_id,        // id of expr/pat producing this value
             span: span,              // span of same expr/pat
             cat: categorization,     // categorization of expr
             lp: Option<@loan_path>,  // loan path for expr, if any
             mutbl: ast::mutability,  // mutability of expr as lvalue
             ty: ty::t};              // type of the expr

type cmt = @cmt_;

impl cmt_ : cmp::Eq {
    pure fn eq(&self, other: &cmt_) -> bool {
        (*self).id == (*other).id &&
        (*self).span == (*other).span &&
        (*self).cat == (*other).cat &&
        (*self).lp == (*other).lp &&
        (*self).mutbl == (*other).mutbl &&
        (*self).ty == (*other).ty
    }
    pure fn ne(&self, other: &cmt_) -> bool { !(*self).eq(other) }
}

// a loan path is like a category, but it exists only when the data is
// interior to the stack frame.  loan paths are used as the key to a
// map indicating what is borrowed at any point in time.
pub enum loan_path {
    lp_local(ast::node_id),
    lp_arg(ast::node_id),
    lp_deref(@loan_path, ptr_kind),
    lp_comp(@loan_path, comp_kind)
}

impl loan_path : cmp::Eq {
    pure fn eq(&self, other: &loan_path) -> bool {
        match (*self) {
            lp_local(e0a) => {
                match (*other) {
                    lp_local(e0b) => e0a == e0b,
                    _ => false
                }
            }
            lp_arg(e0a) => {
                match (*other) {
                    lp_arg(e0b) => e0a == e0b,
                    _ => false
                }
            }
            lp_deref(e0a, e1a) => {
                match (*other) {
                    lp_deref(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            lp_comp(e0a, e1a) => {
                match (*other) {
                    lp_comp(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
        }
    }
    pure fn ne(&self, other: &loan_path) -> bool { !(*self).eq(other) }
}

// We pun on *T to mean both actual deref of a ptr as well
// as accessing of components:
enum deref_kind {deref_ptr(ptr_kind), deref_comp(comp_kind)}

// Categorizes a derefable type.  Note that we include vectors and strings as
// derefable (we model an index as the combination of a deref and then a
// pointer adjustment).
fn opt_deref_kind(t: ty::t) -> Option<deref_kind> {
    match ty::get(t).sty {
      ty::ty_uniq(*) |
      ty::ty_evec(_, ty::vstore_uniq) |
      ty::ty_estr(ty::vstore_uniq) => {
        Some(deref_ptr(uniq_ptr))
      }

      ty::ty_fn(ref f) if (*f).meta.proto == ast::ProtoUniq => {
        Some(deref_ptr(uniq_ptr))
      }

      ty::ty_rptr(r, _) |
      ty::ty_evec(_, ty::vstore_slice(r)) |
      ty::ty_estr(ty::vstore_slice(r)) => {
        Some(deref_ptr(region_ptr(r)))
      }

      ty::ty_fn(ref f) if (*f).meta.proto == ast::ProtoBorrowed => {
        Some(deref_ptr(region_ptr((*f).meta.region)))
      }

      ty::ty_box(*) |
      ty::ty_evec(_, ty::vstore_box) |
      ty::ty_estr(ty::vstore_box) => {
        Some(deref_ptr(gc_ptr))
      }

      ty::ty_fn(ref f) if (*f).meta.proto == ast::ProtoBox => {
        Some(deref_ptr(gc_ptr))
      }

      ty::ty_ptr(*) => {
        Some(deref_ptr(unsafe_ptr))
      }

      ty::ty_enum(did, _) => {
        Some(deref_comp(comp_variant(did)))
      }

      ty::ty_struct(_, _) => {
        Some(deref_comp(comp_anon_field))
      }

      ty::ty_evec(mt, ty::vstore_fixed(_)) => {
        Some(deref_comp(comp_index(t, mt.mutbl)))
      }

      ty::ty_estr(ty::vstore_fixed(_)) => {
        Some(deref_comp(comp_index(t, m_imm)))
      }

      _ => None
    }
}

fn deref_kind(tcx: ty::ctxt, t: ty::t) -> deref_kind {
    match opt_deref_kind(t) {
      Some(k) => k,
      None => {
        tcx.sess.bug(
            fmt!("deref_cat() invoked on non-derefable type %s",
                 ty_to_str(tcx, t)));
      }
    }
}

fn cat_expr(
    tcx: ty::ctxt,
    method_map: typeck::method_map,
    expr: @ast::expr) -> cmt {

    let mcx = &mem_categorization_ctxt {
        tcx: tcx, method_map: method_map
    };
    return mcx.cat_expr(expr);
}

fn cat_expr_unadjusted(
    tcx: ty::ctxt,
    method_map: typeck::method_map,
    expr: @ast::expr) -> cmt {

    let mcx = &mem_categorization_ctxt {
        tcx: tcx, method_map: method_map
    };
    return mcx.cat_expr_unadjusted(expr);
}

fn cat_expr_autoderefd(
    tcx: ty::ctxt,
    method_map: typeck::method_map,
    expr: @ast::expr,
    adj: @ty::AutoAdjustment) -> cmt {

    let mcx = &mem_categorization_ctxt {
        tcx: tcx, method_map: method_map
    };
    return mcx.cat_expr_autoderefd(expr, adj);
}

fn cat_def(
    tcx: ty::ctxt,
    method_map: typeck::method_map,
    expr_id: ast::node_id,
    expr_span: span,
    expr_ty: ty::t,
    def: ast::def) -> cmt {

    let mcx = &mem_categorization_ctxt {
        tcx: tcx, method_map: method_map
    };
    return mcx.cat_def(expr_id, expr_span, expr_ty, def);
}

fn cat_variant<N: ast_node>(
    tcx: ty::ctxt,
    method_map: typeck::method_map,
    arg: N,
    enum_did: ast::def_id,
    cmt: cmt) -> cmt {

    let mcx = &mem_categorization_ctxt {
        tcx: tcx, method_map: method_map
    };
    return mcx.cat_variant(arg, enum_did, cmt);
}

trait ast_node {
    fn id() -> ast::node_id;
    fn span() -> span;
}

impl @ast::expr: ast_node {
    fn id() -> ast::node_id { self.id }
    fn span() -> span { self.span }
}

impl @ast::pat: ast_node {
    fn id() -> ast::node_id { self.id }
    fn span() -> span { self.span }
}

trait get_type_for_node {
    fn ty<N: ast_node>(node: N) -> ty::t;
}

impl ty::ctxt: get_type_for_node {
    fn ty<N: ast_node>(node: N) -> ty::t {
        ty::node_id_to_type(self, node.id())
    }
}

struct mem_categorization_ctxt {
    tcx: ty::ctxt,
    method_map: typeck::method_map,
}

impl &mem_categorization_ctxt {
    fn cat_expr(expr: @ast::expr) -> cmt {
        match self.tcx.adjustments.find(expr.id) {
            None => {
                // No adjustments.
                self.cat_expr_unadjusted(expr)
            }

            Some(adjustment) => {
                match adjustment.autoref {
                    Some(_) => {
                        // Equivalent to &*expr or something similar.
                        // This is an rvalue, effectively.
                        let expr_ty = ty::expr_ty(self.tcx, expr);
                        self.cat_rvalue(expr, expr_ty)
                    }
                    None => {
                        // Equivalent to *expr or something similar.
                        self.cat_expr_autoderefd(expr, adjustment)
                    }
                }
            }
        }
    }

    fn cat_expr_autoderefd(expr: @ast::expr,
                           adjustment: &ty::AutoAdjustment) -> cmt {
        let mut cmt = self.cat_expr_unadjusted(expr);
        for uint::range(1, adjustment.autoderefs+1) |deref| {
            cmt = self.cat_deref(expr, cmt, deref);
        }
        return cmt;
    }

    fn cat_expr_unadjusted(expr: @ast::expr) -> cmt {
        debug!("cat_expr: id=%d expr=%s",
               expr.id, pprust::expr_to_str(expr, self.tcx.sess.intr()));

        let tcx = self.tcx;
        let expr_ty = tcx.ty(expr);
        match expr.node {
          ast::expr_unary(ast::deref, e_base) => {
            if self.method_map.contains_key(expr.id) {
                return self.cat_rvalue(expr, expr_ty);
            }

            let base_cmt = self.cat_expr(e_base);
            self.cat_deref(expr, base_cmt, 0)
          }

          ast::expr_field(base, f_name, _) => {
            if self.method_map.contains_key(expr.id) {
                return self.cat_method_ref(expr, expr_ty);
            }

            let base_cmt = self.cat_expr(base);
            self.cat_field(expr, base_cmt, f_name, expr.id)
          }

          ast::expr_index(base, _) => {
            if self.method_map.contains_key(expr.id) {
                return self.cat_rvalue(expr, expr_ty);
            }

            let base_cmt = self.cat_expr(base);
            self.cat_index(expr, base_cmt)
          }

          ast::expr_path(_) => {
            let def = self.tcx.def_map.get(expr.id);
            self.cat_def(expr.id, expr.span, expr_ty, def)
          }

          ast::expr_paren(e) => self.cat_expr_unadjusted(e),

          ast::expr_addr_of(*) | ast::expr_call(*) |
          ast::expr_swap(*) | ast::expr_assign(*) |
          ast::expr_assign_op(*) | ast::expr_fn(*) | ast::expr_fn_block(*) |
          ast::expr_assert(*) | ast::expr_ret(*) |
          ast::expr_loop_body(*) | ast::expr_do_body(*) |
          ast::expr_unary(*) | ast::expr_method_call(*) |
          ast::expr_copy(*) | ast::expr_cast(*) | ast::expr_fail(*) |
          ast::expr_vstore(*) | ast::expr_vec(*) | ast::expr_tup(*) |
          ast::expr_if(*) | ast::expr_log(*) |
          ast::expr_binary(*) | ast::expr_while(*) |
          ast::expr_block(*) | ast::expr_loop(*) | ast::expr_match(*) |
          ast::expr_lit(*) | ast::expr_break(*) | ast::expr_mac(*) |
          ast::expr_again(*) | ast::expr_rec(*) | ast::expr_struct(*) |
          ast::expr_unary_move(*) | ast::expr_repeat(*) => {
            return self.cat_rvalue(expr, expr_ty);
          }
        }
    }

    fn cat_def(id: ast::node_id,
               span: span,
               expr_ty: ty::t,
               def: ast::def) -> cmt {
        match def {
          ast::def_fn(*) | ast::def_static_method(*) | ast::def_mod(_) |
          ast::def_foreign_mod(_) | ast::def_const(_) |
          ast::def_use(_) | ast::def_variant(*) |
          ast::def_ty(_) | ast::def_prim_ty(_) |
          ast::def_ty_param(*) | ast::def_struct(*) |
          ast::def_typaram_binder(*) | ast::def_region(_) |
          ast::def_label(_) | ast::def_self_ty(*) => {
            @{id:id, span:span,
              cat:cat_special(sk_static_item), lp:None,
              mutbl:m_imm, ty:expr_ty}
          }

          ast::def_arg(vid, mode) => {
            // Idea: make this could be rewritten to model by-ref
            // stuff as `&const` and `&mut`?

            // m: mutability of the argument
            // lp: loan path, must be none for aliasable things
            let {m,lp} = match ty::resolved_mode(self.tcx, mode) {
              ast::by_move | ast::by_copy => {
                {m: m_imm, lp: Some(@lp_arg(vid))}
              }
              ast::by_ref => {
                {m: m_imm, lp: None}
              }
              ast::by_val => {
                // by-value is this hybrid mode where we have a
                // pointer but we do not own it.  This is not
                // considered loanable because, for example, a by-ref
                // and and by-val argument might both actually contain
                // the same unique ptr.
                {m: m_imm, lp: None}
              }
            };
            @{id:id, span:span,
              cat:cat_arg(vid), lp:lp,
              mutbl:m, ty:expr_ty}
          }

          ast::def_self(_, is_implicit) => {
            let special_kind = if is_implicit {
                sk_implicit_self
            } else {
                sk_self
            };

            @{id:id, span:span,
              cat:cat_special(special_kind), lp:None,
              mutbl:m_imm, ty:expr_ty}
          }

          ast::def_upvar(_, inner, fn_node_id, _) => {
            let ty = ty::node_id_to_type(self.tcx, fn_node_id);
            let proto = ty::ty_fn_proto(ty);
            match proto {
                ast::ProtoBorrowed => {
                    let upcmt = self.cat_def(id, span, expr_ty, *inner);
                    @{id:id, span:span,
                      cat:cat_stack_upvar(upcmt), lp:upcmt.lp,
                      mutbl:upcmt.mutbl, ty:upcmt.ty}
                }
                ast::ProtoUniq | ast::ProtoBox => {
                    // FIXME #2152 allow mutation of moved upvars
                    @{id:id, span:span,
                      cat:cat_special(sk_heap_upvar), lp:None,
                      mutbl:m_imm, ty:expr_ty}
                }
                ast::ProtoBare => {
                    self.tcx.sess.span_bug(
                        span,
                        fmt!("Upvar in a bare closure?"));
                }
            }
          }

          ast::def_local(vid, mutbl) => {
            let m = if mutbl {m_mutbl} else {m_imm};
            @{id:id, span:span,
              cat:cat_local(vid), lp:Some(@lp_local(vid)),
              mutbl:m, ty:expr_ty}
          }

          ast::def_binding(vid, _) => {
            // by-value/by-ref bindings are local variables
            @{id:id, span:span,
              cat:cat_local(vid), lp:Some(@lp_local(vid)),
              mutbl:m_imm, ty:expr_ty}
          }
        }
    }

    fn cat_variant<N: ast_node>(arg: N,
                                enum_did: ast::def_id,
                                cmt: cmt) -> cmt {
        @{id: arg.id(), span: arg.span(),
          cat: cat_comp(cmt, comp_variant(enum_did)),
          lp: cmt.lp.map(|l| @lp_comp(*l, comp_variant(enum_did)) ),
          mutbl: cmt.mutbl, // imm iff in an immutable context
          ty: self.tcx.ty(arg)}
    }

    fn cat_rvalue(expr: @ast::expr, expr_ty: ty::t) -> cmt {
        @{id:expr.id, span:expr.span,
          cat:cat_rvalue, lp:None,
          mutbl:m_imm, ty:expr_ty}
    }

    /// inherited mutability: used in cases where the mutability of a
    /// component is inherited from the base it is a part of. For
    /// example, a record field is mutable if it is declared mutable
    /// or if the container is mutable.
    fn inherited_mutability(base_m: ast::mutability,
                          comp_m: ast::mutability) -> ast::mutability {
        match comp_m {
          m_imm => {base_m}  // imm: as mutable as the container
          m_mutbl | m_const => {comp_m}
        }
    }

    /// The `field_id` parameter is the ID of the enclosing expression or
    /// pattern. It is used to determine which variant of an enum is in use.
    fn cat_field<N:ast_node>(node: N,
                             base_cmt: cmt,
                             f_name: ast::ident,
                             field_id: ast::node_id) -> cmt {
        let f_mutbl = match field_mutbl(self.tcx, base_cmt.ty,
                                        f_name, field_id) {
            Some(f_mutbl) => f_mutbl,
            None => {
                self.tcx.sess.span_bug(
                    node.span(),
                    fmt!("Cannot find field `%s` in type `%s`",
                         self.tcx.sess.str_of(f_name),
                         ty_to_str(self.tcx, base_cmt.ty)));
            }
        };
        let m = self.inherited_mutability(base_cmt.mutbl, f_mutbl);
        let f_comp = comp_field(f_name, f_mutbl);
        let lp = base_cmt.lp.map(|lp| @lp_comp(*lp, f_comp) );
        @{id: node.id(), span: node.span(),
          cat: cat_comp(base_cmt, f_comp), lp:lp,
          mutbl: m, ty: self.tcx.ty(node)}
    }

    fn cat_deref_fn<N:ast_node>(node: N,
                                base_cmt: cmt,
                                deref_cnt: uint) -> cmt
    {
        // Bit of a hack: the "dereference" of a function pointer like
        // `@fn()` is a mere logical concept. We interpret it as
        // dereferencing the environment pointer; of course, we don't
        // know what type lies at the other end, so we just call it
        // `()` (the empty tuple).

        let mt = ty::mt {ty: ty::mk_tup(self.tcx, ~[]), mutbl: m_imm};
        return self.cat_deref_common(node, base_cmt, deref_cnt, mt);
    }

    fn cat_deref<N:ast_node>(node: N,
                             base_cmt: cmt,
                             deref_cnt: uint) -> cmt
    {
        let mt = match ty::deref(self.tcx, base_cmt.ty, true) {
            Some(mt) => mt,
            None => {
                self.tcx.sess.span_bug(
                    node.span(),
                    fmt!("Explicit deref of non-derefable type: %s",
                         ty_to_str(self.tcx, base_cmt.ty)));
            }
        };

        return self.cat_deref_common(node, base_cmt, deref_cnt, mt);
    }

    fn cat_deref_common<N:ast_node>(node: N,
                                    base_cmt: cmt,
                                    deref_cnt: uint,
                                    mt: ty::mt) -> cmt
    {
        match deref_kind(self.tcx, base_cmt.ty) {
            deref_ptr(ptr) => {
                let lp = do base_cmt.lp.chain_ref |l| {
                    // Given that the ptr itself is loanable, we can
                    // loan out deref'd uniq ptrs as the data they are
                    // the only way to reach the data they point at.
                    // Other ptr types admit aliases and are therefore
                    // not loanable.
                    match ptr {
                        uniq_ptr => {Some(@lp_deref(*l, ptr))}
                        gc_ptr | region_ptr(_) | unsafe_ptr => {None}
                    }
                };

                // for unique ptrs, we inherit mutability from the
                // owning reference.
                let m = match ptr {
                    uniq_ptr => {
                        self.inherited_mutability(base_cmt.mutbl, mt.mutbl)
                    }
                    gc_ptr | region_ptr(_) | unsafe_ptr => {
                        mt.mutbl
                    }
                };

                @{id:node.id(), span:node.span(),
                  cat:cat_deref(base_cmt, deref_cnt, ptr), lp:lp,
                  mutbl:m, ty:mt.ty}
            }

            deref_comp(comp) => {
                let lp = base_cmt.lp.map(|l| @lp_comp(*l, comp) );
                let m = self.inherited_mutability(base_cmt.mutbl, mt.mutbl);
                @{id:node.id(), span:node.span(),
                  cat:cat_comp(base_cmt, comp), lp:lp,
                  mutbl:m, ty:mt.ty}
            }
        }
    }

    fn cat_index(expr: @ast::expr, base_cmt: cmt) -> cmt {
        let mt = match ty::index(self.tcx, base_cmt.ty) {
          Some(mt) => mt,
          None => {
            self.tcx.sess.span_bug(
                expr.span,
                fmt!("Explicit index of non-index type `%s`",
                     ty_to_str(self.tcx, base_cmt.ty)));
          }
        };

        return match deref_kind(self.tcx, base_cmt.ty) {
          deref_ptr(ptr) => {
            // (a) the contents are loanable if the base is loanable
            // and this is a *unique* vector
            let deref_lp = match ptr {
              uniq_ptr => {base_cmt.lp.map(|lp| @lp_deref(*lp, uniq_ptr))}
              _ => {None}
            };

            // (b) for unique ptrs, we inherit mutability from the
            // owning reference.
            let m = match ptr {
              uniq_ptr => {
                self.inherited_mutability(base_cmt.mutbl, mt.mutbl)
              }
              gc_ptr | region_ptr(_) | unsafe_ptr => {
                mt.mutbl
              }
            };

            // (c) the deref is explicit in the resulting cmt
            let deref_cmt = @{id:expr.id, span:expr.span,
              cat:cat_deref(base_cmt, 0u, ptr), lp:deref_lp,
              mutbl:m, ty:mt.ty};

            comp(expr, deref_cmt, base_cmt.ty, m, mt.ty)
          }

          deref_comp(_) => {
            // fixed-length vectors have no deref
            let m = self.inherited_mutability(base_cmt.mutbl, mt.mutbl);
            comp(expr, base_cmt, base_cmt.ty, m, mt.ty)
          }
        };

        fn comp(expr: @ast::expr, of_cmt: cmt,
                vect: ty::t, mutbl: ast::mutability, ty: ty::t) -> cmt {
            let comp = comp_index(vect, mutbl);
            let index_lp = of_cmt.lp.map(|lp| @lp_comp(*lp, comp) );
            @{id:expr.id, span:expr.span,
              cat:cat_comp(of_cmt, comp), lp:index_lp,
              mutbl:mutbl, ty:ty}
        }
    }

    fn cat_tuple_elt<N: ast_node>(elt: N, cmt: cmt) -> cmt {
        @{id: elt.id(), span: elt.span(),
          cat: cat_comp(cmt, comp_tuple),
          lp: cmt.lp.map(|l| @lp_comp(*l, comp_tuple) ),
          mutbl: cmt.mutbl, // imm iff in an immutable context
          ty: self.tcx.ty(elt)}
    }

    fn cat_anon_struct_field<N: ast_node>(elt: N, cmt: cmt) -> cmt {
        @{id: elt.id(), span: elt.span(),
          cat: cat_comp(cmt, comp_anon_field),
          lp: cmt.lp.map(|l| @lp_comp(*l, comp_anon_field)),
          mutbl: cmt.mutbl, // imm iff in an immutable context
          ty: self.tcx.ty(elt)}
    }

    fn cat_method_ref(expr: @ast::expr, expr_ty: ty::t) -> cmt {
        @{id:expr.id, span:expr.span,
          cat:cat_special(sk_method), lp:None,
          mutbl:m_imm, ty:expr_ty}
    }

    fn cat_pattern(cmt: cmt, pat: @ast::pat, op: fn(cmt, @ast::pat)) {

        op(cmt, pat);

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
        // The correspondence between the id in the cmt and which
        // pattern is being referred to is somewhat...subtle.  In
        // general, the id of the cmt is the id of the node that
        // produces the value.  For patterns, that's actually the
        // *subpattern*, generally speaking.
        //
        // To see what I mean about ids etc, consider:
        //
        //     let x = @@3;
        //     match x {
        //       @@y { ... }
        //     }
        //
        // Here the cmt for `y` would be something like
        //
        //     local(x)->@->@
        //
        // where the id of `local(x)` is the id of the `x` that appears
        // in the match, the id of `local(x)->@` is the `@y` pattern,
        // and the id of `local(x)->@->@` is the id of the `y` pattern.


        let _i = indenter();
        let tcx = self.tcx;
        debug!("cat_pattern: id=%d pat=%s cmt=%s",
               pat.id, pprust::pat_to_str(pat, tcx.sess.intr()),
               self.cmt_to_repr(cmt));

        match /*bad*/copy pat.node {
          ast::pat_wild => {
            // _
          }

          ast::pat_enum(_, None) => {
            // variant(*)
          }
          ast::pat_enum(_, Some(subpats)) => {
            match self.tcx.def_map.find(pat.id) {
                Some(ast::def_variant(enum_did, _)) => {
                    // variant(x, y, z)
                    for subpats.each |subpat| {
                        let subcmt = self.cat_variant(*subpat, enum_did, cmt);
                        self.cat_pattern(subcmt, *subpat, op);
                    }
                }
                Some(ast::def_struct(*)) => {
                    for subpats.each |subpat| {
                        let cmt_field = self.cat_anon_struct_field(*subpat,
                                                                   cmt);
                        self.cat_pattern(cmt_field, *subpat, op);
                    }
                }
                _ => {
                    self.tcx.sess.span_bug(
                        pat.span,
                        ~"enum pattern didn't resolve to enum or struct");
                }
            }
          }

          ast::pat_ident(_, _, Some(subpat)) => {
              self.cat_pattern(cmt, subpat, op);
          }

          ast::pat_ident(_, _, None) => {
              // nullary variant or identifier: ignore
          }

          ast::pat_rec(field_pats, _) => {
            // {f1: p1, ..., fN: pN}
            for field_pats.each |fp| {
                let cmt_field = self.cat_field(fp.pat, cmt, fp.ident, pat.id);
                self.cat_pattern(cmt_field, fp.pat, op);
            }
          }

          ast::pat_struct(_, field_pats, _) => {
            // {f1: p1, ..., fN: pN}
            for field_pats.each |fp| {
                let cmt_field = self.cat_field(fp.pat, cmt, fp.ident, pat.id);
                self.cat_pattern(cmt_field, fp.pat, op);
            }
          }

          ast::pat_tup(subpats) => {
            // (p1, ..., pN)
            for subpats.each |subpat| {
                let subcmt = self.cat_tuple_elt(*subpat, cmt);
                self.cat_pattern(subcmt, *subpat, op);
            }
          }

          ast::pat_box(subpat) | ast::pat_uniq(subpat) |
          ast::pat_region(subpat) => {
            // @p1, ~p1
            let subcmt = self.cat_deref(subpat, cmt, 0);
            self.cat_pattern(subcmt, subpat, op);
          }

          ast::pat_vec(*) | ast::pat_lit(_) | ast::pat_range(_, _) => {
              /*always ok*/
          }
        }
    }

    fn cat_to_repr(cat: categorization) -> ~str {
        match cat {
          cat_special(sk_method) => ~"method",
          cat_special(sk_static_item) => ~"static_item",
          cat_special(sk_implicit_self) => ~"implicit-self",
          cat_special(sk_self) => ~"self",
          cat_special(sk_heap_upvar) => ~"heap-upvar",
          cat_stack_upvar(_) => ~"stack-upvar",
          cat_rvalue => ~"rvalue",
          cat_local(node_id) => fmt!("local(%d)", node_id),
          cat_binding(node_id) => fmt!("binding(%d)", node_id),
          cat_arg(node_id) => fmt!("arg(%d)", node_id),
          cat_deref(cmt, derefs, ptr) => {
            fmt!("%s->(%s, %u)", self.cat_to_repr(cmt.cat),
                 self.ptr_sigil(ptr), derefs)
          }
          cat_comp(cmt, comp) => {
            fmt!("%s.%s", self.cat_to_repr(cmt.cat), self.comp_to_repr(comp))
          }
          cat_discr(cmt, _) => self.cat_to_repr(cmt.cat)
        }
    }

    fn mut_to_str(mutbl: ast::mutability) -> ~str {
        match mutbl {
          m_mutbl => ~"mutable",
          m_const => ~"const",
          m_imm => ~"immutable"
        }
    }

    fn ptr_sigil(ptr: ptr_kind) -> ~str {
        match ptr {
          uniq_ptr => ~"~",
          gc_ptr => ~"@",
          region_ptr(_) => ~"&",
          unsafe_ptr => ~"*"
        }
    }

    fn comp_to_repr(comp: comp_kind) -> ~str {
        match comp {
          comp_field(fld, _) => self.tcx.sess.str_of(fld),
          comp_index(*) => ~"[]",
          comp_tuple => ~"()",
          comp_anon_field => ~"<anonymous field>",
          comp_variant(_) => ~"<enum>"
        }
    }

    fn lp_to_str(lp: @loan_path) -> ~str {
        match *lp {
          lp_local(node_id) => {
            fmt!("local(%d)", node_id)
          }
          lp_arg(node_id) => {
            fmt!("arg(%d)", node_id)
          }
          lp_deref(lp, ptr) => {
            fmt!("%s->(%s)", self.lp_to_str(lp),
                 self.ptr_sigil(ptr))
          }
          lp_comp(lp, comp) => {
            fmt!("%s.%s", self.lp_to_str(lp),
                 self.comp_to_repr(comp))
          }
        }
    }

    fn cmt_to_repr(cmt: cmt) -> ~str {
        fmt!("{%s id:%d m:%s lp:%s ty:%s}",
             self.cat_to_repr(cmt.cat),
             cmt.id,
             self.mut_to_str(cmt.mutbl),
             cmt.lp.map_default(~"none", |p| self.lp_to_str(*p) ),
             ty_to_str(self.tcx, cmt.ty))
    }

    fn cmt_to_str(cmt: cmt) -> ~str {
        let mut_str = self.mut_to_str(cmt.mutbl);
        match cmt.cat {
          cat_special(sk_method) => ~"method",
          cat_special(sk_static_item) => ~"static item",
          cat_special(sk_implicit_self) => ~"self reference",
          cat_special(sk_self) => ~"self value",
          cat_special(sk_heap_upvar) => {
              ~"captured outer variable in a heap closure"
          }
          cat_rvalue => ~"non-lvalue",
          cat_local(_) => mut_str + ~" local variable",
          cat_binding(_) => ~"pattern binding",
          cat_arg(_) => ~"argument",
          cat_deref(_, _, pk) => fmt!("dereference of %s %s pointer",
                                      mut_str, self.ptr_sigil(pk)),
          cat_stack_upvar(_) => {
            ~"captured outer " + mut_str + ~" variable in a stack closure"
          }
          cat_comp(_, comp_field(*)) => mut_str + ~" field",
          cat_comp(_, comp_tuple) => ~"tuple content",
          cat_comp(_, comp_anon_field) => ~"anonymous field",
          cat_comp(_, comp_variant(_)) => ~"enum content",
          cat_comp(_, comp_index(t, _)) => {
            match ty::get(t).sty {
              ty::ty_evec(*) => mut_str + ~" vec content",
              ty::ty_estr(*) => mut_str + ~" str content",
              _ => mut_str + ~" indexed content"
            }
          }
          cat_discr(cmt, _) => {
            self.cmt_to_str(cmt)
          }
        }
    }

    fn region_to_str(r: ty::Region) -> ~str {
        region_to_str(self.tcx, r)
    }
}

/// The node_id here is the node of the expression that references the field.
/// This function looks it up in the def map in case the type happens to be
/// an enum to determine which variant is in use.
fn field_mutbl(tcx: ty::ctxt,
               base_ty: ty::t,
               f_name: ast::ident,
               node_id: ast::node_id) -> Option<ast::mutability> {
    // Need to refactor so that records/class fields can be treated uniformly.
    match /*bad*/copy ty::get(base_ty).sty {
      ty::ty_rec(fields) => {
        for fields.each |f| {
            if f.ident == f_name {
                return Some(f.mt.mutbl);
            }
        }
      }
      ty::ty_struct(did, _) => {
        for ty::lookup_struct_fields(tcx, did).each |fld| {
            if fld.ident == f_name {
                let m = match fld.mutability {
                  ast::struct_mutable => ast::m_mutbl,
                  ast::struct_immutable => ast::m_imm
                };
                return Some(m);
            }
        }
      }
      ty::ty_enum(*) => {
        match tcx.def_map.get(node_id) {
          ast::def_variant(_, variant_id) => {
            for ty::lookup_struct_fields(tcx, variant_id).each |fld| {
                if fld.ident == f_name {
                    let m = match fld.mutability {
                      ast::struct_mutable => ast::m_mutbl,
                      ast::struct_immutable => ast::m_imm
                    };
                    return Some(m);
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
