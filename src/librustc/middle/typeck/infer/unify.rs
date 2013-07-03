// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use extra::smallintmap::SmallIntMap;

use middle::ty::{Vid, expected_found, IntVarValue};
use middle::ty;
use middle::typeck::infer::{Bounds, uok, ures};
use middle::typeck::infer::InferCtxt;
use middle::typeck::infer::to_str::InferStr;
use syntax::ast;

pub enum VarValue<V, T> {
    Redirect(V),
    Root(T, uint),
}

pub struct ValsAndBindings<V, T> {
    vals: SmallIntMap<VarValue<V, T>>,
    bindings: ~[(V, VarValue<V, T>)],
}

pub struct Node<V, T> {
    root: V,
    possible_types: T,
    rank: uint,
}

pub trait UnifyVid<T> {
    fn appropriate_vals_and_bindings<'v>(infcx: &'v mut InferCtxt)
                                      -> &'v mut ValsAndBindings<Self, T>;
}

pub trait UnifyInferCtxtMethods {
    fn get<T:Copy,
           V:Copy + Eq + Vid + UnifyVid<T>>(
           &mut self,
           vid: V)
           -> Node<V, T>;
    fn set<T:Copy + InferStr,
           V:Copy + Vid + ToStr + UnifyVid<T>>(
           &mut self,
           vid: V,
           new_v: VarValue<V, T>);
    fn unify<T:Copy + InferStr,
             V:Copy + Vid + ToStr + UnifyVid<T>>(
             &mut self,
             node_a: &Node<V, T>,
             node_b: &Node<V, T>)
             -> (V, uint);
}

impl UnifyInferCtxtMethods for InferCtxt {
    fn get<T:Copy,
           V:Copy + Eq + Vid + UnifyVid<T>>(
           &mut self,
           vid: V)
           -> Node<V, T> {
        /*!
         *
         * Find the root node for `vid`. This uses the standard
         * union-find algorithm with path compression:
         * http://en.wikipedia.org/wiki/Disjoint-set_data_structure
         */

        let tcx = self.tcx;
        let vb = UnifyVid::appropriate_vals_and_bindings(self);
        return helper(tcx, vb, vid);

        fn helper<T:Copy, V:Copy+Eq+Vid>(
            tcx: ty::ctxt,
            vb: &mut ValsAndBindings<V,T>,
            vid: V) -> Node<V, T>
        {
            let vid_u = vid.to_uint();
            let var_val = match vb.vals.find(&vid_u) {
                Some(&ref var_val) => copy *var_val,
                None => {
                    tcx.sess.bug(fmt!(
                        "failed lookup of vid `%u`", vid_u));
                }
            };
            match var_val {
                Redirect(vid) => {
                    let node: Node<V,T> = helper(tcx, vb, copy vid);
                    if node.root != vid {
                        // Path compression
                        vb.vals.insert(vid.to_uint(),
                                       Redirect(copy node.root));
                    }
                    node
                }
                Root(pt, rk) => {
                    Node {root: vid, possible_types: pt, rank: rk}
                }
            }
        }
    }

    fn set<T:Copy + InferStr,
           V:Copy + Vid + ToStr + UnifyVid<T>>(
           &mut self,
           vid: V,
           new_v: VarValue<V, T>) {
        /*!
         *
         * Sets the value for `vid` to `new_v`.  `vid` MUST be a root node!
         */

        debug!("Updating variable %s to %s",
               vid.to_str(), new_v.inf_str(self));

        let vb = UnifyVid::appropriate_vals_and_bindings(self);
        let old_v = copy *vb.vals.get(&vid.to_uint());
        vb.bindings.push((copy vid, old_v));
        vb.vals.insert(vid.to_uint(), new_v);
    }

    fn unify<T:Copy + InferStr,
             V:Copy + Vid + ToStr + UnifyVid<T>>(
             &mut self,
             node_a: &Node<V, T>,
             node_b: &Node<V, T>)
             -> (V, uint) {
        // Rank optimization: if you don't know what it is, check
        // out <http://en.wikipedia.org/wiki/Disjoint-set_data_structure>

        debug!("unify(node_a(id=%?, rank=%?), \
                node_b(id=%?, rank=%?))",
               node_a.root, node_a.rank,
               node_b.root, node_b.rank);

        if node_a.rank > node_b.rank {
            // a has greater rank, so a should become b's parent,
            // i.e., b should redirect to a.
            self.set(copy node_b.root, Redirect(copy node_a.root));
            (copy node_a.root, node_a.rank)
        } else if node_a.rank < node_b.rank {
            // b has greater rank, so a should redirect to b.
            self.set(copy node_a.root, Redirect(copy node_b.root));
            (copy node_b.root, node_b.rank)
        } else {
            // If equal, redirect one to the other and increment the
            // other's rank.
            assert_eq!(node_a.rank, node_b.rank);
            self.set(copy node_b.root, Redirect(copy node_a.root));
            (copy node_a.root, node_a.rank + 1)
        }
    }

}

// ______________________________________________________________________
// Code to handle simple variables like ints, floats---anything that
// doesn't have a subtyping relationship we need to worry about.

pub trait SimplyUnifiable {
    fn to_type_err(expected_found<Self>) -> ty::type_err;
}

pub fn mk_err<T:SimplyUnifiable>(a_is_expected: bool,
                                 a_t: T,
                                 b_t: T) -> ures {
    if a_is_expected {
        Err(SimplyUnifiable::to_type_err(
            ty::expected_found {expected: a_t, found: b_t}))
    } else {
        Err(SimplyUnifiable::to_type_err(
            ty::expected_found {expected: b_t, found: a_t}))
    }
}

pub trait InferCtxtMethods {
    fn simple_vars<T:Copy + Eq + InferStr + SimplyUnifiable,
                   V:Copy + Eq + Vid + ToStr + UnifyVid<Option<T>>>(
                   &mut self,
                   a_is_expected: bool,
                   a_id: V,
                   b_id: V)
                   -> ures;
    fn simple_var_t<T:Copy + Eq + InferStr + SimplyUnifiable,
                    V:Copy + Eq + Vid + ToStr + UnifyVid<Option<T>>>(
                    &mut self,
                    a_is_expected: bool,
                    a_id: V,
                    b: T)
                    -> ures;
}

impl InferCtxtMethods for InferCtxt {
    fn simple_vars<T:Copy + Eq + InferStr + SimplyUnifiable,
                   V:Copy + Eq + Vid + ToStr + UnifyVid<Option<T>>>(
                   &mut self,
                   a_is_expected: bool,
                   a_id: V,
                   b_id: V)
                   -> ures {
        /*!
         *
         * Unifies two simple variables.  Because simple variables do
         * not have any subtyping relationships, if both variables
         * have already been associated with a value, then those two
         * values must be the same. */

        let node_a = self.get(a_id);
        let node_b = self.get(b_id);
        let a_id = copy node_a.root;
        let b_id = copy node_b.root;

        if a_id == b_id { return uok(); }

        let combined = match (&node_a.possible_types, &node_b.possible_types)
        {
            (&None, &None) => None,
            (&Some(ref v), &None) | (&None, &Some(ref v)) => Some(copy *v),
            (&Some(ref v1), &Some(ref v2)) => {
                if *v1 != *v2 {
                    return mk_err(a_is_expected, copy *v1, copy *v2);
                }
                Some(copy *v1)
            }
        };

        let (new_root, new_rank) = self.unify(&node_a, &node_b);
        self.set(new_root, Root(combined, new_rank));
        return uok();
    }

    fn simple_var_t<T:Copy + Eq + InferStr + SimplyUnifiable,
                    V:Copy + Eq + Vid + ToStr + UnifyVid<Option<T>>>(
                    &mut self,
                    a_is_expected: bool,
                    a_id: V,
                    b: T)
                    -> ures {
        /*!
         *
         * Sets the value of the variable `a_id` to `b`.  Because
         * simple variables do not have any subtyping relationships,
         * if `a_id` already has a value, it must be the same as
         * `b`. */

        let node_a = self.get(a_id);
        let a_id = copy node_a.root;

        match node_a.possible_types {
            None => {
                self.set(a_id, Root(Some(b), node_a.rank));
                return uok();
            }

            Some(ref a_t) => {
                if *a_t == b {
                    return uok();
                } else {
                    return mk_err(a_is_expected, copy *a_t, b);
                }
            }
        }
    }
}

// ______________________________________________________________________

impl UnifyVid<Bounds<ty::t>> for ty::TyVid {
    fn appropriate_vals_and_bindings<'v>(infcx: &'v mut InferCtxt)
        -> &'v mut ValsAndBindings<ty::TyVid, Bounds<ty::t>> {
        return &mut infcx.ty_var_bindings;
    }
}

impl UnifyVid<Option<IntVarValue>> for ty::IntVid {
    fn appropriate_vals_and_bindings<'v>(infcx: &'v mut InferCtxt)
        -> &'v mut ValsAndBindings<ty::IntVid, Option<IntVarValue>> {
        return &mut infcx.int_var_bindings;
    }
}

impl SimplyUnifiable for IntVarValue {
    fn to_type_err(err: expected_found<IntVarValue>) -> ty::type_err {
        return ty::terr_int_mismatch(err);
    }
}

impl UnifyVid<Option<ast::float_ty>> for ty::FloatVid {
    fn appropriate_vals_and_bindings<'v>(infcx: &'v mut InferCtxt)
        -> &'v mut ValsAndBindings<ty::FloatVid, Option<ast::float_ty>> {
        return &mut infcx.float_var_bindings;
    }
}

impl SimplyUnifiable for ast::float_ty {
    fn to_type_err(err: expected_found<ast::float_ty>) -> ty::type_err {
        return ty::terr_float_mismatch(err);
    }
}
