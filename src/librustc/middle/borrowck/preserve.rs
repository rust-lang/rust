// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ----------------------------------------------------------------------
// Preserve(Ex, S) holds if ToAddr(Ex) will remain valid for the entirety of
// the scope S.
//

use core::prelude::*;

use middle::borrowck::{RootInfo, bckerr, bckerr_code, bckres, borrowck_ctxt};
use middle::borrowck::{cmt, err_mut_uniq, err_mut_variant};
use middle::borrowck::{err_out_of_root_scope, err_out_of_scope};
use middle::borrowck::{err_root_not_permitted};
use middle::mem_categorization::{cat_arg, cat_binding, cat_comp, cat_deref};
use middle::mem_categorization::{cat_discr, cat_local, cat_self, cat_special};
use middle::mem_categorization::{cat_stack_upvar, comp_field, comp_index};
use middle::mem_categorization::{comp_variant, gc_ptr, region_ptr};
use middle::ty;
use util::common::indenter;

use syntax::ast::{m_const, m_imm, m_mutbl};
use syntax::ast;

export public_methods, preserve_condition, pc_ok, pc_if_pure;

enum preserve_condition {
    pc_ok,
    pc_if_pure(bckerr)
}

impl preserve_condition {
    // combines two preservation conditions such that if either of
    // them requires purity, the result requires purity
    fn combine(pc: preserve_condition) -> preserve_condition {
        match self {
          pc_ok => {pc}
          pc_if_pure(_) => {self}
        }
    }
}

impl borrowck_ctxt {
    fn preserve(cmt: cmt,
                scope_region: ty::Region,
                item_ub: ast::node_id,
                root_ub: ast::node_id)
        -> bckres<preserve_condition> {

        let ctxt = preserve_ctxt({bccx: self,
                                  scope_region: scope_region,
                                  item_ub: item_ub,
                                  root_ub: root_ub,
                                  root_managed_data: true});
        (&ctxt).preserve(cmt)
    }
}

enum preserve_ctxt = {
    bccx: borrowck_ctxt,

    // the region scope for which we must preserve the memory
    scope_region: ty::Region,

    // the scope for the body of the enclosing fn/method item
    item_ub: ast::node_id,

    // the upper bound on how long we can root an @T pointer
    root_ub: ast::node_id,

    // if false, do not attempt to root managed data
    root_managed_data: bool
};


priv impl &preserve_ctxt {
    fn tcx() -> ty::ctxt { self.bccx.tcx }

    fn preserve(cmt: cmt) -> bckres<preserve_condition> {
        debug!("preserve(cmt=%s, root_ub=%?, root_managed_data=%b)",
               self.bccx.cmt_to_repr(cmt), self.root_ub,
               self.root_managed_data);
        let _i = indenter();

        match cmt.cat {
          cat_special(sk_implicit_self) |
          cat_special(sk_heap_upvar) => {
            self.compare_scope(cmt, ty::re_scope(self.item_ub))
          }
          cat_special(sk_static_item) | cat_special(sk_method) => {
            Ok(pc_ok)
          }
          cat_rvalue => {
            // when we borrow an rvalue, we can keep it rooted but only
            // up to the root_ub point

            // When we're in a 'const &x = ...' context, self.root_ub is
            // zero and the rvalue is static, not bound to a scope.
            let scope_region = if self.root_ub == 0 {
                ty::re_static
            } else {
                // Maybe if we pass in the parent instead here,
                // we can prevent the "scope not found" error
                debug!("scope_region thing: %? ", cmt.id);
                ty::re_scope(self.tcx().region_map.get(cmt.id))
            };

            self.compare_scope(cmt, scope_region)
          }
          cat_stack_upvar(cmt) => {
            self.preserve(cmt)
          }
          cat_local(local_id) => {
            // Normally, local variables are lendable, and so this
            // case should never trigger.  However, if we are
            // preserving an expression like a.b where the field `b`
            // has @ type, then it will recurse to ensure that the `a`
            // is stable to try and avoid rooting the value `a.b`.  In
            // this case, root_managed_data will be false.
            if self.root_managed_data {
                self.tcx().sess.span_bug(
                    cmt.span,
                    ~"preserve() called with local and !root_managed_data");
            }
            let local_scope_id = self.tcx().region_map.get(local_id);
            self.compare_scope(cmt, ty::re_scope(local_scope_id))
          }
          cat_binding(local_id) => {
            // Bindings are these kind of weird implicit pointers (cc
            // #2329).  We require (in gather_loans) that they be
            // rooted in an immutable location.
            let local_scope_id = self.tcx().region_map.get(local_id);
            self.compare_scope(cmt, ty::re_scope(local_scope_id))
          }
          cat_arg(local_id) => {
            // This can happen as not all args are lendable (e.g., &&
            // modes).  In that case, the caller guarantees stability
            // for at least the scope of the fn.  This is basically a
            // deref of a region ptr.
            let local_scope_id = self.tcx().region_map.get(local_id);
            self.compare_scope(cmt, ty::re_scope(local_scope_id))
          }
          cat_self(local_id) => {
            let local_scope_id = self.tcx().region_map.get(local_id);
            self.compare_scope(cmt, ty::re_scope(local_scope_id))
          }
          cat_comp(cmt_base, comp_field(*)) |
          cat_comp(cmt_base, comp_index(*)) |
          cat_comp(cmt_base, comp_tuple) |
          cat_comp(cmt_base, comp_anon_field) => {
            // Most embedded components: if the base is stable, the
            // type never changes.
            self.preserve(cmt_base)
          }
          cat_comp(cmt_base, comp_variant(enum_did)) => {
            if ty::enum_is_univariant(self.tcx(), enum_did) {
                self.preserve(cmt_base)
            } else {
                // If there are multiple variants: overwriting the
                // base could cause the type of this memory to change,
                // so require imm.
                self.require_imm(cmt, cmt_base, err_mut_variant)
            }
          }
          cat_deref(cmt_base, _, uniq_ptr) => {
            // Overwriting the base could cause this memory to be
            // freed, so require imm.
            self.require_imm(cmt, cmt_base, err_mut_uniq)
          }
          cat_deref(_, _, region_ptr(_, region)) => {
            // References are always "stable" for lifetime `region` by
            // induction (when the reference of type &MT was created,
            // the memory must have been stable).
            self.compare_scope(cmt, region)
          }
          cat_deref(_, _, unsafe_ptr) => {
            // Unsafe pointers are the user's problem
            Ok(pc_ok)
          }
          cat_deref(base, derefs, gc_ptr(*)) => {
            // GC'd pointers of type @MT: if this pointer lives in
            // immutable, stable memory, then everything is fine.  But
            // otherwise we have no guarantee the pointer will stay
            // live, so we must root the pointer (i.e., inc the ref
            // count) for the duration of the loan.
            debug!("base.mutbl = %?", self.bccx.mut_to_str(base.mutbl));
            if cmt.cat.derefs_through_mutable_box() {
                self.attempt_root(cmt, base, derefs)
            } else if base.mutbl == m_imm {
                let non_rooting_ctxt =
                    preserve_ctxt({root_managed_data: false,.. **self});
                match (&non_rooting_ctxt).preserve(base) {
                  Ok(pc_ok) => {
                    Ok(pc_ok)
                  }
                  Ok(pc_if_pure(_)) => {
                    debug!("must root @T, otherwise purity req'd");
                    self.attempt_root(cmt, base, derefs)
                  }
                  Err(ref e) => {
                    debug!("must root @T, err: %s",
                           self.bccx.bckerr_to_str((*e)));
                    self.attempt_root(cmt, base, derefs)
                  }
                }
            } else {
                self.attempt_root(cmt, base, derefs)
            }
          }
          cat_discr(base, match_id) => {
            // Subtle: in a match, we must ensure that each binding
            // variable remains valid for the duration of the arm in
            // which it appears, presuming that this arm is taken.
            // But it is inconvenient in trans to root something just
            // for one arm.  Therefore, we insert a cat_discr(),
            // basically a special kind of category that says "if this
            // value must be dynamically rooted, root it for the scope
            // `match_id`.
            //
            // As an example, consider this scenario:
            //
            //    let mut x = @some(3);
            //    match *x { Some(y) {...} none {...} }
            //
            // Technically, the value `x` need only be rooted
            // in the `some` arm.  However, we evaluate `x` in trans
            // before we know what arm will be taken, so we just
            // always root it for the duration of the match.
            //
            // As a second example, consider *this* scenario:
            //
            //    let x = @mut @some(3);
            //    match x { @@some(y) {...} @@none {...} }
            //
            // Here again, `x` need only be rooted in the `some` arm.
            // In this case, the value which needs to be rooted is
            // found only when checking which pattern matches: but
            // this check is done before entering the arm.  Therefore,
            // even in this case we just choose to keep the value
            // rooted for the entire match.  This means the value will be
            // rooted even if the none arm is taken.  Oh well.
            //
            // At first, I tried to optimize the second case to only
            // root in one arm, but the result was suboptimal: first,
            // it interfered with the construction of phi nodes in the
            // arm, as we were adding code to root values before the
            // phi nodes were added.  This could have been addressed
            // with a second basic block.  However, the naive approach
            // also yielded suboptimal results for patterns like:
            //
            //    let x = @mut @...;
            //    match x { @@some_variant(y) | @@some_other_variant(y) =>
            //
            // The reason is that we would root the value once for
            // each pattern and not once per arm.  This is also easily
            // fixed, but it's yet more code for what is really quite
            // the corner case.
            //
            // Nonetheless, if you decide to optimize this case in the
            // future, you need only adjust where the cat_discr()
            // node appears to draw the line between what will be rooted
            // in the *arm* vs the *match*.

            let match_rooting_ctxt =
                preserve_ctxt({scope_region: ty::re_scope(match_id),
                               .. **self});
            (&match_rooting_ctxt).preserve(base)
          }
        }
    }

    /// Reqiures that `cmt` (which is a deref or subcomponent of
    /// `base`) be found in an immutable location (that is, `base`
    /// must be immutable).  Also requires that `base` itself is
    /// preserved.
    fn require_imm(cmt: cmt,
                   cmt_base: cmt,
                   code: bckerr_code) -> bckres<preserve_condition> {
        // Variant contents and unique pointers: must be immutably
        // rooted to a preserved address.
        match self.preserve(cmt_base) {
          // the base is preserved, but if we are not mutable then
          // purity is required
          Ok(pc_ok) => {
            match cmt_base.mutbl {
              m_mutbl | m_const => {
                Ok(pc_if_pure({cmt:cmt, code:code}))
              }
              m_imm => {
                Ok(pc_ok)
              }
            }
          }

          // the base requires purity too, that's fine
          Ok(pc_if_pure(ref e)) => {
            Ok(pc_if_pure((*e)))
          }

          // base is not stable, doesn't matter
          Err(ref e) => {
            Err((*e))
          }
        }
    }

    /// Checks that the scope for which the value must be preserved
    /// is a subscope of `scope_ub`; if so, success.
    fn compare_scope(cmt: cmt,
                     scope_ub: ty::Region) -> bckres<preserve_condition> {
        if self.bccx.is_subregion_of(self.scope_region, scope_ub) {
            Ok(pc_ok)
        } else {
            Err({cmt:cmt, code:err_out_of_scope(scope_ub,
                                                self.scope_region)})
        }
    }

    /// Here, `cmt=*base` is always a deref of managed data (if
    /// `derefs` != 0, then an auto-deref).  This routine determines
    /// whether it is safe to MAKE cmt stable by rooting the pointer
    /// `base`.  We can only do the dynamic root if the desired
    /// lifetime `self.scope_region` is a subset of `self.root_ub`
    /// scope; otherwise, it would either require that we hold the
    /// value live for longer than the current fn or else potentially
    /// require that an statically unbounded number of values be
    /// rooted (if a loop exists).
    fn attempt_root(cmt: cmt,
                    base: cmt,
                    derefs: uint)
                 -> bckres<preserve_condition> {
        if !self.root_managed_data {
            // normally, there is a root_ub; the only time that this
            // is none is when a boxed value is stored in an immutable
            // location.  In that case, we will test to see if that
            // immutable location itself can be preserved long enough
            // in which case no rooting is necessary.  But there it
            // would be sort of pointless to avoid rooting the inner
            // box by rooting an outer box, as it would just keep more
            // memory live than necessary, so we set root_ub to none.
            return Err({cmt:cmt, code:err_root_not_permitted});
        }

        let root_region = ty::re_scope(self.root_ub);
        match self.scope_region {
          // we can only root values if the desired region is some concrete
          // scope within the fn body
          ty::re_scope(scope_id) => {
            debug!("Considering root map entry for %s: \
                    node %d:%u -> scope_id %?, root_ub %?",
                   self.bccx.cmt_to_repr(cmt), base.id,
                   derefs, scope_id, self.root_ub);
            if self.bccx.is_subregion_of(self.scope_region, root_region) {
                debug!("Elected to root");
                let rk = {id: base.id, derefs: derefs};
                // This code could potentially lead cause boxes to be frozen
                // for longer than necessarily at runtime. It prevents an
                // ICE in trans; the fundamental problem is that it's hard
                // to make sure trans and borrowck have the same notion of
                // scope. The real fix is to clean up how trans handles
                // cleanups, but that's hard. If this becomes an issue, it's
                // an option to just change this to `let scope_to_use =
                // scope_id;`. Though that would potentially re-introduce
                // the ICE. See #3511 for more details.
                let scope_to_use = if
                    self.bccx.stmt_map.contains_key(scope_id) {
                    // Root it in its parent scope, b/c
                    // trans won't introduce a new scope for the
                    // stmt
                    self.root_ub
                }
                else {
                    // Use the more precise scope
                    scope_id
                };
                // We freeze if and only if this is a *mutable* @ box that
                // we're borrowing into a pointer.
                self.bccx.root_map.insert(rk, RootInfo {
                    scope: scope_to_use,
                    freezes: cmt.cat.derefs_through_mutable_box()
                });
                return Ok(pc_ok);
            } else {
                debug!("Unable to root");
                return Err({cmt:cmt,
                            code:err_out_of_root_scope(root_region,
                                                       self.scope_region)});
            }
          }

          // we won't be able to root long enough
          _ => {
              return Err({cmt:cmt,
                          code:err_out_of_root_scope(root_region,
                                                     self.scope_region)});
          }

        }
    }
}
