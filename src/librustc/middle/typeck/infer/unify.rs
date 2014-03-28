// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use collections::SmallIntMap;

use middle::ty::{Vid, expected_found, IntVarValue};
use middle::ty;
use middle::typeck::infer::{Bounds, uok, ures};
use middle::typeck::infer::InferCtxt;
use middle::typeck::infer::to_str::InferStr;
use std::cell::RefCell;
use syntax::ast;

#[deriving(Clone)]
pub enum VarValue<V, T> {
    Redirect(V),
    Root(T, uint),
}

pub struct ValsAndBindings<V, T> {
    pub vals: SmallIntMap<VarValue<V, T>>,
    pub bindings: Vec<(V, VarValue<V, T>)> ,
}

pub struct Node<V, T> {
    pub root: V,
    pub possible_types: T,
    pub rank: uint,
}

pub trait UnifyVid<T> {
    fn appropriate_vals_and_bindings<'v>(infcx: &'v InferCtxt)
                                     -> &'v RefCell<ValsAndBindings<Self, T>>;
}

pub trait UnifyInferCtxtMethods {
    fn get<T:Clone,
           V:Clone + Eq + Vid + UnifyVid<T>>(
           &self,
           vid: V)
           -> Node<V, T>;
    fn set<T:Clone + InferStr,
           V:Clone + Vid + ToStr + UnifyVid<T>>(
           &self,
           vid: V,
           new_v: VarValue<V, T>);
    fn unify<T:Clone + InferStr,
             V:Clone + Vid + ToStr + UnifyVid<T>>(
             &self,
             node_a: &Node<V, T>,
             node_b: &Node<V, T>)
             -> (V, uint);
}

impl<'a> UnifyInferCtxtMethods for InferCtxt<'a> {
    fn get<T:Clone,
           V:Clone + Eq + Vid + UnifyVid<T>>(
           &self,
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
        return helper(tcx, &mut *vb.borrow_mut(), vid);

        fn helper<T:Clone, V:Clone+Eq+Vid>(
            tcx: &ty::ctxt,
            vb: &mut ValsAndBindings<V,T>,
            vid: V) -> Node<V, T>
        {
            let vid_u = vid.to_uint();
            let var_val = match vb.vals.find(&vid_u) {
                Some(&ref var_val) => (*var_val).clone(),
                None => {
                    tcx.sess.bug(format!(
                        "failed lookup of vid `{}`", vid_u));
                }
            };
            match var_val {
                Redirect(vid) => {
                    let node: Node<V,T> = helper(tcx, vb, vid.clone());
                    if node.root != vid {
                        // Path compression
                        vb.vals.insert(vid.to_uint(),
                                       Redirect(node.root.clone()));
                    }
                    node
                }
                Root(pt, rk) => {
                    Node {root: vid, possible_types: pt, rank: rk}
                }
            }
        }
    }

    fn set<T:Clone + InferStr,
           V:Clone + Vid + ToStr + UnifyVid<T>>(
           &self,
           vid: V,
           new_v: VarValue<V, T>) {
        /*!
         *
         * Sets the value for `vid` to `new_v`.  `vid` MUST be a root node!
         */

        debug!("Updating variable {} to {}",
               vid.to_str(), new_v.inf_str(self));

        let vb = UnifyVid::appropriate_vals_and_bindings(self);
        let mut vb = vb.borrow_mut();
        let old_v = (*vb.vals.get(&vid.to_uint())).clone();
        vb.bindings.push((vid.clone(), old_v));
        vb.vals.insert(vid.to_uint(), new_v);
    }

    fn unify<T:Clone + InferStr,
             V:Clone + Vid + ToStr + UnifyVid<T>>(
             &self,
             node_a: &Node<V, T>,
             node_b: &Node<V, T>)
             -> (V, uint) {
        // Rank optimization: if you don't know what it is, check
        // out <http://en.wikipedia.org/wiki/Disjoint-set_data_structure>

        debug!("unify(node_a(id={:?}, rank={:?}), \
                node_b(id={:?}, rank={:?}))",
               node_a.root, node_a.rank,
               node_b.root, node_b.rank);

        if node_a.rank > node_b.rank {
            // a has greater rank, so a should become b's parent,
            // i.e., b should redirect to a.
            self.set(node_b.root.clone(), Redirect(node_a.root.clone()));
            (node_a.root.clone(), node_a.rank)
        } else if node_a.rank < node_b.rank {
            // b has greater rank, so a should redirect to b.
            self.set(node_a.root.clone(), Redirect(node_b.root.clone()));
            (node_b.root.clone(), node_b.rank)
        } else {
            // If equal, redirect one to the other and increment the
            // other's rank.
            assert_eq!(node_a.rank, node_b.rank);
            self.set(node_b.root.clone(), Redirect(node_a.root.clone()));
            (node_a.root.clone(), node_a.rank + 1)
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
    fn simple_vars<T:Clone + Eq + InferStr + SimplyUnifiable,
                   V:Clone + Eq + Vid + ToStr + UnifyVid<Option<T>>>(
                   &self,
                   a_is_expected: bool,
                   a_id: V,
                   b_id: V)
                   -> ures;
    fn simple_var_t<T:Clone + Eq + InferStr + SimplyUnifiable,
                    V:Clone + Eq + Vid + ToStr + UnifyVid<Option<T>>>(
                    &self,
                    a_is_expected: bool,
                    a_id: V,
                    b: T)
                    -> ures;
}

impl<'a> InferCtxtMethods for InferCtxt<'a> {
    fn simple_vars<T:Clone + Eq + InferStr + SimplyUnifiable,
                   V:Clone + Eq + Vid + ToStr + UnifyVid<Option<T>>>(
                   &self,
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
        let a_id = node_a.root.clone();
        let b_id = node_b.root.clone();

        if a_id == b_id { return uok(); }

        let combined = match (&node_a.possible_types, &node_b.possible_types)
        {
            (&None, &None) => None,
            (&Some(ref v), &None) | (&None, &Some(ref v)) => {
                Some((*v).clone())
            }
            (&Some(ref v1), &Some(ref v2)) => {
                if *v1 != *v2 {
                    return mk_err(a_is_expected, (*v1).clone(), (*v2).clone())
                }
                Some((*v1).clone())
            }
        };

        let (new_root, new_rank) = self.unify(&node_a, &node_b);
        self.set(new_root, Root(combined, new_rank));
        return uok();
    }

    fn simple_var_t<T:Clone + Eq + InferStr + SimplyUnifiable,
                    V:Clone + Eq + Vid + ToStr + UnifyVid<Option<T>>>(
                    &self,
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
        let a_id = node_a.root.clone();

        match node_a.possible_types {
            None => {
                self.set(a_id, Root(Some(b), node_a.rank));
                return uok();
            }

            Some(ref a_t) => {
                if *a_t == b {
                    return uok();
                } else {
                    return mk_err(a_is_expected, (*a_t).clone(), b);
                }
            }
        }
    }
}

// ______________________________________________________________________

impl UnifyVid<Bounds<ty::t>> for ty::TyVid {
    fn appropriate_vals_and_bindings<'v>(infcx: &'v InferCtxt)
        -> &'v RefCell<ValsAndBindings<ty::TyVid, Bounds<ty::t>>> {
        return &infcx.ty_var_bindings;
    }
}

impl UnifyVid<Option<IntVarValue>> for ty::IntVid {
    fn appropriate_vals_and_bindings<'v>(infcx: &'v InferCtxt)
        -> &'v RefCell<ValsAndBindings<ty::IntVid, Option<IntVarValue>>> {
        return &infcx.int_var_bindings;
    }
}

impl SimplyUnifiable for IntVarValue {
    fn to_type_err(err: expected_found<IntVarValue>) -> ty::type_err {
        return ty::terr_int_mismatch(err);
    }
}

impl UnifyVid<Option<ast::FloatTy>> for ty::FloatVid {
    fn appropriate_vals_and_bindings<'v>(infcx: &'v InferCtxt)
        -> &'v RefCell<ValsAndBindings<ty::FloatVid, Option<ast::FloatTy>>> {
        return &infcx.float_var_bindings;
    }
}

impl SimplyUnifiable for ast::FloatTy {
    fn to_type_err(err: expected_found<ast::FloatTy>) -> ty::type_err {
        return ty::terr_float_mismatch(err);
    }
}
