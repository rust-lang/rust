// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::kinds::marker;

use middle::ty::{expected_found, IntVarValue};
use middle::ty;
use middle::typeck::infer::{uok, ures};
use middle::typeck::infer::InferCtxt;
use std::cell::RefCell;
use std::fmt::Show;
use syntax::ast;
use util::ppaux::Repr;
use util::snapshot_vec as sv;

/**
 * This trait is implemented by any type that can serve as a type
 * variable. We call such variables *unification keys*. For example,
 * this trait is implemented by `IntVid`, which represents integral
 * variables.
 *
 * Each key type has an associated value type `V`. For example, for
 * `IntVid`, this is `Option<IntVarValue>`, representing some
 * (possibly not yet known) sort of integer.
 *
 * Implementations of this trait are at the end of this file.
 */
pub trait UnifyKey<V> : Clone + Show + PartialEq + Repr {
    fn index(&self) -> uint;

    fn from_index(u: uint) -> Self;

    /**
     * Given an inference context, returns the unification table
     * appropriate to this key type.
     */
    fn unification_table<'v>(infcx: &'v InferCtxt)
                             -> &'v RefCell<UnificationTable<Self,V>>;

    fn tag(k: Option<Self>) -> &'static str;
}

/**
 * Trait for valid types that a type variable can be set to. Note that
 * this is typically not the end type that the value will take on, but
 * rather an `Option` wrapper (where `None` represents a variable
 * whose value is not yet set).
 *
 * Implementations of this trait are at the end of this file.
 */
pub trait UnifyValue : Clone + Repr + PartialEq {
}

/**
 * Value of a unification key. We implement Tarjan's union-find
 * algorithm: when two keys are unified, one of them is converted
 * into a "redirect" pointing at the other. These redirects form a
 * DAG: the roots of the DAG (nodes that are not redirected) are each
 * associated with a value of type `V` and a rank. The rank is used
 * to keep the DAG relatively balanced, which helps keep the running
 * time of the algorithm under control. For more information, see
 * <http://en.wikipedia.org/wiki/Disjoint-set_data_structure>.
 */
#[deriving(PartialEq,Clone)]
pub enum VarValue<K,V> {
    Redirect(K),
    Root(V, uint),
}

/**
 * Table of unification keys and their values.
 */
pub struct UnificationTable<K,V> {
    /**
     * Indicates the current value of each key.
     */

    values: sv::SnapshotVec<VarValue<K,V>,(),Delegate>,
}

/**
 * At any time, users may snapshot a unification table.  The changes
 * made during the snapshot may either be *committed* or *rolled back*.
 */
pub struct Snapshot<K> {
    // Link snapshot to the key type `K` of the table.
    marker: marker::CovariantType<K>,
    snapshot: sv::Snapshot,
}

/**
 * Internal type used to represent the result of a `get()` operation.
 * Conveys the current root and value of the key.
 */
pub struct Node<K,V> {
    pub key: K,
    pub value: V,
    pub rank: uint,
}

pub struct Delegate;

// We can't use V:LatticeValue, much as I would like to,
// because frequently the pattern is that V=Option<U> for some
// other type parameter U, and we have no way to say
// Option<U>:LatticeValue.

impl<V:PartialEq+Clone+Repr,K:UnifyKey<V>> UnificationTable<K,V> {
    pub fn new() -> UnificationTable<K,V> {
        UnificationTable {
            values: sv::SnapshotVec::new(Delegate),
        }
    }

    /**
     * Starts a new snapshot. Each snapshot must be either
     * rolled back or committed in a "LIFO" (stack) order.
     */
    pub fn snapshot(&mut self) -> Snapshot<K> {
        Snapshot { marker: marker::CovariantType::<K>,
                   snapshot: self.values.start_snapshot() }
    }

    /**
     * Reverses all changes since the last snapshot. Also
     * removes any keys that have been created since then.
     */
    pub fn rollback_to(&mut self, snapshot: Snapshot<K>) {
        debug!("{}: rollback_to()", UnifyKey::tag(None::<K>));
        self.values.rollback_to(snapshot.snapshot);
    }

    /**
     * Commits all changes since the last snapshot. Of course, they
     * can still be undone if there is a snapshot further out.
     */
    pub fn commit(&mut self, snapshot: Snapshot<K>) {
        debug!("{}: commit()", UnifyKey::tag(None::<K>));
        self.values.commit(snapshot.snapshot);
    }

    pub fn new_key(&mut self, value: V) -> K {
        let index = self.values.push(Root(value, 0));
        let k = UnifyKey::from_index(index);
        debug!("{}: created new key: {}",
               UnifyKey::tag(None::<K>),
               k);
        k
    }

    pub fn get(&mut self, tcx: &ty::ctxt, vid: K) -> Node<K,V> {
        /*!
         * Find the root node for `vid`. This uses the standard
         * union-find algorithm with path compression:
         * http://en.wikipedia.org/wiki/Disjoint-set_data_structure
         */

        let index = vid.index();
        let value = (*self.values.get(index)).clone();
        match value {
            Redirect(redirect) => {
                let node: Node<K,V> = self.get(tcx, redirect.clone());
                if node.key != redirect {
                    // Path compression
                    self.values.set(index, Redirect(node.key.clone()));
                }
                node
            }
            Root(value, rank) => {
                Node { key: vid, value: value, rank: rank }
            }
        }
    }

    fn is_root(&self, key: &K) -> bool {
        match *self.values.get(key.index()) {
            Redirect(..) => false,
            Root(..) => true,
        }
    }

    pub fn set(&mut self,
               tcx: &ty::ctxt,
               key: K,
               new_value: VarValue<K,V>)
    {
        /*!
         * Sets the value for `vid` to `new_value`. `vid` MUST be a
         * root node! Also, we must be in the middle of a snapshot.
         */

        assert!(self.is_root(&key));

        debug!("Updating variable {} to {}",
               key.repr(tcx),
               new_value.repr(tcx));

        self.values.set(key.index(), new_value);
    }

    pub fn unify(&mut self,
                 tcx: &ty::ctxt,
                 node_a: &Node<K,V>,
                 node_b: &Node<K,V>)
                 -> (K, uint)
    {
        /*!
         * Either redirects node_a to node_b or vice versa, depending
         * on the relative rank. Returns the new root and rank.  You
         * should then update the value of the new root to something
         * suitable.
         */

        debug!("unify(node_a(id={}, rank={}), node_b(id={}, rank={}))",
               node_a.key.repr(tcx),
               node_a.rank,
               node_b.key.repr(tcx),
               node_b.rank);

        if node_a.rank > node_b.rank {
            // a has greater rank, so a should become b's parent,
            // i.e., b should redirect to a.
            self.set(tcx, node_b.key.clone(), Redirect(node_a.key.clone()));
            (node_a.key.clone(), node_a.rank)
        } else if node_a.rank < node_b.rank {
            // b has greater rank, so a should redirect to b.
            self.set(tcx, node_a.key.clone(), Redirect(node_b.key.clone()));
            (node_b.key.clone(), node_b.rank)
        } else {
            // If equal, redirect one to the other and increment the
            // other's rank.
            assert_eq!(node_a.rank, node_b.rank);
            self.set(tcx, node_b.key.clone(), Redirect(node_a.key.clone()));
            (node_a.key.clone(), node_a.rank + 1)
        }
    }
}

impl<K,V> sv::SnapshotVecDelegate<VarValue<K,V>,()> for Delegate {
    fn reverse(&mut self, _: &mut Vec<VarValue<K,V>>, _: ()) {
        fail!("Nothing to reverse");
    }
}

///////////////////////////////////////////////////////////////////////////
// Code to handle simple keys like ints, floats---anything that
// doesn't have a subtyping relationship we need to worry about.

/**
 * Indicates a type that does not have any kind of subtyping
 * relationship.
 */
pub trait SimplyUnifiable : Clone + PartialEq + Repr {
    fn to_type_err(expected_found<Self>) -> ty::type_err;
}

pub fn err<V:SimplyUnifiable>(a_is_expected: bool,
                              a_t: V,
                              b_t: V)
                              -> ures {
    if a_is_expected {
        Err(SimplyUnifiable::to_type_err(
            ty::expected_found {expected: a_t, found: b_t}))
    } else {
        Err(SimplyUnifiable::to_type_err(
            ty::expected_found {expected: b_t, found: a_t}))
    }
}

pub trait InferCtxtMethodsForSimplyUnifiableTypes<V:SimplyUnifiable,
                                                  K:UnifyKey<Option<V>>> {
    fn simple_vars(&self,
                   a_is_expected: bool,
                   a_id: K,
                   b_id: K)
                   -> ures;
    fn simple_var_t(&self,
                    a_is_expected: bool,
                    a_id: K,
                    b: V)
                    -> ures;
}

impl<'a,'tcx,V:SimplyUnifiable,K:UnifyKey<Option<V>>>
    InferCtxtMethodsForSimplyUnifiableTypes<V,K> for InferCtxt<'a, 'tcx>
{
    fn simple_vars(&self,
                   a_is_expected: bool,
                   a_id: K,
                   b_id: K)
                   -> ures
    {
        /*!
         * Unifies two simple keys.  Because simple keys do
         * not have any subtyping relationships, if both keys
         * have already been associated with a value, then those two
         * values must be the same.
         */

        let tcx = self.tcx;
        let table = UnifyKey::unification_table(self);
        let node_a = table.borrow_mut().get(tcx, a_id);
        let node_b = table.borrow_mut().get(tcx, b_id);
        let a_id = node_a.key.clone();
        let b_id = node_b.key.clone();

        if a_id == b_id { return uok(); }

        let combined = {
            match (&node_a.value, &node_b.value) {
                (&None, &None) => {
                    None
                }
                (&Some(ref v), &None) | (&None, &Some(ref v)) => {
                    Some((*v).clone())
                }
                (&Some(ref v1), &Some(ref v2)) => {
                    if *v1 != *v2 {
                        return err(a_is_expected, (*v1).clone(), (*v2).clone())
                    }
                    Some((*v1).clone())
                }
            }
        };

        let (new_root, new_rank) = table.borrow_mut().unify(tcx,
                                                            &node_a,
                                                            &node_b);
        table.borrow_mut().set(tcx, new_root, Root(combined, new_rank));
        return Ok(())
    }

    fn simple_var_t(&self,
                    a_is_expected: bool,
                    a_id: K,
                    b: V)
                    -> ures
    {
        /*!
         * Sets the value of the key `a_id` to `b`.  Because
         * simple keys do not have any subtyping relationships,
         * if `a_id` already has a value, it must be the same as
         * `b`.
         */

        let tcx = self.tcx;
        let table = UnifyKey::unification_table(self);
        let node_a = table.borrow_mut().get(tcx, a_id);
        let a_id = node_a.key.clone();

        match node_a.value {
            None => {
                table.borrow_mut().set(tcx, a_id, Root(Some(b), node_a.rank));
                return Ok(());
            }

            Some(ref a_t) => {
                if *a_t == b {
                    return Ok(());
                } else {
                    return err(a_is_expected, (*a_t).clone(), b);
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////

// Integral type keys

impl UnifyKey<Option<IntVarValue>> for ty::IntVid {
    fn index(&self) -> uint { self.index }

    fn from_index(i: uint) -> ty::IntVid { ty::IntVid { index: i } }

    fn unification_table<'v>(infcx: &'v InferCtxt)
        -> &'v RefCell<UnificationTable<ty::IntVid, Option<IntVarValue>>>
    {
        return &infcx.int_unification_table;
    }

    fn tag(_: Option<ty::IntVid>) -> &'static str {
        "IntVid"
    }
}

impl SimplyUnifiable for IntVarValue {
    fn to_type_err(err: expected_found<IntVarValue>) -> ty::type_err {
        return ty::terr_int_mismatch(err);
    }
}

impl UnifyValue for Option<IntVarValue> { }

// Floating point type keys

impl UnifyKey<Option<ast::FloatTy>> for ty::FloatVid {
    fn index(&self) -> uint { self.index }

    fn from_index(i: uint) -> ty::FloatVid { ty::FloatVid { index: i } }

    fn unification_table<'v>(infcx: &'v InferCtxt)
        -> &'v RefCell<UnificationTable<ty::FloatVid, Option<ast::FloatTy>>>
    {
        return &infcx.float_unification_table;
    }

    fn tag(_: Option<ty::FloatVid>) -> &'static str {
        "FloatVid"
    }
}

impl UnifyValue for Option<ast::FloatTy> {
}

impl SimplyUnifiable for ast::FloatTy {
    fn to_type_err(err: expected_found<ast::FloatTy>) -> ty::type_err {
        return ty::terr_float_mismatch(err);
    }
}

impl<K:Repr,V:Repr> Repr for VarValue<K,V> {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        match *self {
            Redirect(ref k) => format!("Redirect({})", k.repr(tcx)),
            Root(ref v, r) => format!("Root({}, {})", v.repr(tcx), r)
        }
    }
}
