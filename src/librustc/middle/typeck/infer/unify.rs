// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use middle::ty::Vid;
use middle::ty;
use middle::typeck::infer::{Bound, Bounds, cres, uok, ures};
use middle::typeck::infer::combine::Combine;
use middle::typeck::infer::InferCtxt;
use middle::typeck::infer::to_str::InferStr;
use util::common::{indent, indenter};

use core::result;
use std::smallintmap::SmallIntMap;

enum VarValue<V, T> {
    Redirect(V),
    Root(T, uint),
}

struct ValsAndBindings<V:Copy, T:Copy> {
    vals: SmallIntMap<VarValue<V, T>>,
    mut bindings: ~[(V, VarValue<V, T>)],
}

struct Node<V:Copy, T:Copy> {
    root: V,
    possible_types: T,
    rank: uint,
}

impl @InferCtxt {
    fn get<V:Copy Eq Vid, T:Copy>(
        vb: &ValsAndBindings<V, T>,
        vid: V)
        -> Node<V, T>
    {
        /*!
         *
         * Find the root node for `vid`. This uses the standard
         * union-find algorithm with path compression:
         * http://en.wikipedia.org/wiki/Disjoint-set_data_structure
         */

        let vid_u = vid.to_uint();
        match vb.vals.find(vid_u) {
          None => {
            self.tcx.sess.bug(fmt!("failed lookup of vid `%u`", vid_u));
          }
          Some(ref var_val) => {
            match (*var_val) {
              Redirect(ref vid) => {
                let node = self.get(vb, (*vid));
                if node.root.ne(vid) {
                    // Path compression
                    vb.vals.insert(vid.to_uint(), Redirect(node.root));
                }
                node
              }
              Root(ref pt, rk) => {
                Node {root: vid, possible_types: *pt, rank: rk}
              }
            }
          }
        }
    }

    fn set<V:Copy Vid ToStr, T:Copy InferStr>(
        vb: &ValsAndBindings<V, T>,
        vid: V,
        +new_v: VarValue<V, T>)
    {
        /*!
         *
         * Sets the value for `vid` to `new_v`.  `vid` MUST be a root node!
         */

        let old_v = vb.vals.get(vid.to_uint());
        vb.bindings.push((vid, old_v));
        vb.vals.insert(vid.to_uint(), new_v);

        debug!("Updating variable %s from %s to %s",
               vid.to_str(), old_v.inf_str(self), new_v.inf_str(self));
    }

    fn unify<V:Copy Vid ToStr, T:Copy InferStr, R>(
        vb: &ValsAndBindings<V, T>,
        node_a: &Node<V, T>,
        node_b: &Node<V, T>,
        op: &fn(new_root: V, new_rank: uint) -> R
    ) -> R {
        // Rank optimization: if you don't know what it is, check
        // out <http://en.wikipedia.org/wiki/Disjoint-set_data_structure>

        debug!("unify(node_a(id=%?, rank=%?), \
                node_b(id=%?, rank=%?))",
               node_a.root, node_a.rank,
               node_b.root, node_b.rank);

        if node_a.rank > node_b.rank {
            // a has greater rank, so a should become b's parent,
            // i.e., b should redirect to a.
            self.set(vb, node_b.root, Redirect(node_a.root));
            op(node_a.root, node_a.rank)
        } else if node_a.rank < node_b.rank {
            // b has greater rank, so a should redirect to b.
            self.set(vb, node_a.root, Redirect(node_b.root));
            op(node_b.root, node_b.rank)
        } else {
            // If equal, redirect one to the other and increment the
            // other's rank.
            assert node_a.rank == node_b.rank;
            self.set(vb, node_b.root, Redirect(node_a.root));
            op(node_a.root, node_a.rank + 1)
        }
    }

}

// ______________________________________________________________________
// Code to handle simple variables like ints, floats---anything that
// doesn't have a subtyping relationship we need to worry about.

impl @InferCtxt {
    fn simple_vars<V:Copy Eq Vid ToStr, T:Copy Eq InferStr>(
        vb: &ValsAndBindings<V, Option<T>>,
        err: ty::type_err,
        a_id: V,
        b_id: V) -> ures
    {
        /*!
         *
         * Unifies two simple variables.  Because simple variables do
         * not have any subtyping relationships, if both variables
         * have already been associated with a value, then those two
         * values must be the same. */

        let node_a = self.get(vb, a_id);
        let node_b = self.get(vb, b_id);
        let a_id = node_a.root;
        let b_id = node_b.root;

        if a_id == b_id { return uok(); }

        let combined = match (&node_a.possible_types, &node_b.possible_types)
        {
            (&None, &None) => None,
            (&Some(ref v), &None) | (&None, &Some(ref v)) => Some(*v),
            (&Some(ref v1), &Some(ref v2)) => {
                if *v1 != *v2 { return Err(err); }
                Some(*v1)
            }
        };

        self.unify(vb, &node_a, &node_b, |new_root, new_rank| {
            self.set(vb, new_root, Root(combined, new_rank));
        });
        return uok();
    }

    fn simple_var_t<V:Copy Eq Vid ToStr, T:Copy Eq InferStr>(
        vb: &ValsAndBindings<V, Option<T>>,
        err: ty::type_err,
        a_id: V,
        b: T) -> ures
    {
        /*!
         *
         * Sets the value of the variable `a_id` to `b`.  Because
         * simple variables do not have any subtyping relationships,
         * if `a_id` already has a value, it must be the same as
         * `b`. */

        let node_a = self.get(vb, a_id);
        let a_id = node_a.root;

        if node_a.possible_types.is_none() {
            self.set(vb, a_id, Root(Some(b), node_a.rank));
            return uok();
        }

        if node_a.possible_types == Some(b) {
            return uok();
        }

        return Err(err);
    }
}

