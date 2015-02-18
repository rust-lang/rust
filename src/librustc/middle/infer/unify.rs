// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::VarValue::*;

use std::marker;

use middle::ty::{expected_found, IntVarValue};
use middle::ty::{self, Ty};
use middle::infer::{uok, ures};
use middle::infer::InferCtxt;
use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
use syntax::ast;
use util::snapshot_vec as sv;

/// This trait is implemented by any type that can serve as a type
/// variable. We call such variables *unification keys*. For example,
/// this trait is implemented by `IntVid`, which represents integral
/// variables.
///
/// Each key type has an associated value type `V`. For example, for
/// `IntVid`, this is `Option<IntVarValue>`, representing some
/// (possibly not yet known) sort of integer.
///
/// Implementations of this trait are at the end of this file.
pub trait UnifyKey : Clone + Debug + PartialEq {
    type Value : UnifyValue;

    fn index(&self) -> uint;

    fn from_index(u: uint) -> Self;

    // Given an inference context, returns the unification table
    // appropriate to this key type.
    fn unification_table<'v>(infcx: &'v InferCtxt)
                             -> &'v RefCell<UnificationTable<Self>>;

    fn tag(k: Option<Self>) -> &'static str;
}

/// Trait for valid types that a type variable can be set to. Note that
/// this is typically not the end type that the value will take on, but
/// rather an `Option` wrapper (where `None` represents a variable
/// whose value is not yet set).
///
/// Implementations of this trait are at the end of this file.
pub trait UnifyValue : Clone + PartialEq + Debug {
}

/// Value of a unification key. We implement Tarjan's union-find
/// algorithm: when two keys are unified, one of them is converted
/// into a "redirect" pointing at the other. These redirects form a
/// DAG: the roots of the DAG (nodes that are not redirected) are each
/// associated with a value of type `V` and a rank. The rank is used
/// to keep the DAG relatively balanced, which helps keep the running
/// time of the algorithm under control. For more information, see
/// <http://en.wikipedia.org/wiki/Disjoint-set_data_structure>.
#[derive(PartialEq,Clone,Debug)]
pub enum VarValue<K:UnifyKey> {
    Redirect(K),
    Root(K::Value, uint),
}

/// Table of unification keys and their values.
pub struct UnificationTable<K:UnifyKey> {
    /// Indicates the current value of each key.
    values: sv::SnapshotVec<Delegate<K>>,
}

/// At any time, users may snapshot a unification table.  The changes
/// made during the snapshot may either be *committed* or *rolled back*.
pub struct Snapshot<K:UnifyKey> {
    // Link snapshot to the key type `K` of the table.
    marker: marker::PhantomData<K>,
    snapshot: sv::Snapshot,
}

/// Internal type used to represent the result of a `get()` operation.
/// Conveys the current root and value of the key.
pub struct Node<K:UnifyKey> {
    pub key: K,
    pub value: K::Value,
    pub rank: uint,
}

#[derive(Copy)]
pub struct Delegate<K>(PhantomData<K>);

// We can't use V:LatticeValue, much as I would like to,
// because frequently the pattern is that V=Option<U> for some
// other type parameter U, and we have no way to say
// Option<U>:LatticeValue.

impl<K:UnifyKey> UnificationTable<K> {
    pub fn new() -> UnificationTable<K> {
        UnificationTable {
            values: sv::SnapshotVec::new(Delegate(PhantomData)),
        }
    }

    /// Starts a new snapshot. Each snapshot must be either
    /// rolled back or committed in a "LIFO" (stack) order.
    pub fn snapshot(&mut self) -> Snapshot<K> {
        Snapshot { marker: marker::PhantomData::<K>,
                   snapshot: self.values.start_snapshot() }
    }

    /// Reverses all changes since the last snapshot. Also
    /// removes any keys that have been created since then.
    pub fn rollback_to(&mut self, snapshot: Snapshot<K>) {
        debug!("{}: rollback_to()", UnifyKey::tag(None::<K>));
        self.values.rollback_to(snapshot.snapshot);
    }

    /// Commits all changes since the last snapshot. Of course, they
    /// can still be undone if there is a snapshot further out.
    pub fn commit(&mut self, snapshot: Snapshot<K>) {
        debug!("{}: commit()", UnifyKey::tag(None::<K>));
        self.values.commit(snapshot.snapshot);
    }

    pub fn new_key(&mut self, value: K::Value) -> K {
        let index = self.values.push(Root(value, 0));
        let k = UnifyKey::from_index(index);
        debug!("{}: created new key: {:?}",
               UnifyKey::tag(None::<K>),
               k);
        k
    }

    /// Find the root node for `vid`. This uses the standard union-find algorithm with path
    /// compression: http://en.wikipedia.org/wiki/Disjoint-set_data_structure
    pub fn get(&mut self, tcx: &ty::ctxt, vid: K) -> Node<K> {
        let index = vid.index();
        let value = (*self.values.get(index)).clone();
        match value {
            Redirect(redirect) => {
                let node: Node<K> = self.get(tcx, redirect.clone());
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

    /// Sets the value for `vid` to `new_value`. `vid` MUST be a root node! Also, we must be in the
    /// middle of a snapshot.
    pub fn set<'tcx>(&mut self,
                     _tcx: &ty::ctxt<'tcx>,
                     key: K,
                     new_value: VarValue<K>)
    {
        assert!(self.is_root(&key));

        debug!("Updating variable {:?} to {:?}",
               key, new_value);

        self.values.set(key.index(), new_value);
    }

    /// Either redirects node_a to node_b or vice versa, depending on the relative rank. Returns
    /// the new root and rank. You should then update the value of the new root to something
    /// suitable.
    pub fn unify<'tcx>(&mut self,
                       tcx: &ty::ctxt<'tcx>,
                       node_a: &Node<K>,
                       node_b: &Node<K>)
                       -> (K, uint)
    {
        debug!("unify(node_a(id={:?}, rank={:?}), node_b(id={:?}, rank={:?}))",
               node_a.key,
               node_a.rank,
               node_b.key,
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

impl<K:UnifyKey> sv::SnapshotVecDelegate for Delegate<K> {
    type Value = VarValue<K>;
    type Undo = ();

    fn reverse(&mut self, _: &mut Vec<VarValue<K>>, _: ()) {
        panic!("Nothing to reverse");
    }
}

///////////////////////////////////////////////////////////////////////////
// Code to handle simple keys like ints, floats---anything that
// doesn't have a subtyping relationship we need to worry about.

/// Indicates a type that does not have any kind of subtyping
/// relationship.
pub trait SimplyUnifiable<'tcx> : Clone + PartialEq + Debug {
    fn to_type(&self, tcx: &ty::ctxt<'tcx>) -> Ty<'tcx>;
    fn to_type_err(expected_found<Self>) -> ty::type_err<'tcx>;
}

pub fn err<'tcx, V:SimplyUnifiable<'tcx>>(a_is_expected: bool,
                                          a_t: V,
                                          b_t: V)
                                          -> ures<'tcx> {
    if a_is_expected {
        Err(SimplyUnifiable::to_type_err(
            ty::expected_found {expected: a_t, found: b_t}))
    } else {
        Err(SimplyUnifiable::to_type_err(
            ty::expected_found {expected: b_t, found: a_t}))
    }
}

pub trait InferCtxtMethodsForSimplyUnifiableTypes<'tcx,K,V>
    where K : UnifyKey<Value=Option<V>>,
          V : SimplyUnifiable<'tcx>,
          Option<V> : UnifyValue,
{
    fn simple_vars(&self,
                   a_is_expected: bool,
                   a_id: K,
                   b_id: K)
                   -> ures<'tcx>;
    fn simple_var_t(&self,
                    a_is_expected: bool,
                    a_id: K,
                    b: V)
                    -> ures<'tcx>;
    fn probe_var(&self, a_id: K) -> Option<Ty<'tcx>>;
}

impl<'a,'tcx,V,K> InferCtxtMethodsForSimplyUnifiableTypes<'tcx,K,V> for InferCtxt<'a,'tcx>
    where K : UnifyKey<Value=Option<V>>,
          V : SimplyUnifiable<'tcx>,
          Option<V> : UnifyValue,
{
    /// Unifies two simple keys. Because simple keys do not have any subtyping relationships, if
    /// both keys have already been associated with a value, then those two values must be the
    /// same.
    fn simple_vars(&self,
                   a_is_expected: bool,
                   a_id: K,
                   b_id: K)
                   -> ures<'tcx>
    {
        let tcx = self.tcx;
        let table = UnifyKey::unification_table(self);
        let node_a: Node<K> = table.borrow_mut().get(tcx, a_id);
        let node_b: Node<K> = table.borrow_mut().get(tcx, b_id);
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

    /// Sets the value of the key `a_id` to `b`. Because simple keys do not have any subtyping
    /// relationships, if `a_id` already has a value, it must be the same as `b`.
    fn simple_var_t(&self,
                    a_is_expected: bool,
                    a_id: K,
                    b: V)
                    -> ures<'tcx>
    {
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

    fn probe_var(&self, a_id: K) -> Option<Ty<'tcx>> {
        let tcx = self.tcx;
        let table = UnifyKey::unification_table(self);
        let node_a = table.borrow_mut().get(tcx, a_id);
        match node_a.value {
            None => None,
            Some(ref a_t) => Some(a_t.to_type(tcx))
        }
    }
}

///////////////////////////////////////////////////////////////////////////

// Integral type keys

impl UnifyKey for ty::IntVid {
    type Value = Option<IntVarValue>;

    fn index(&self) -> uint { self.index as uint }

    fn from_index(i: uint) -> ty::IntVid { ty::IntVid { index: i as u32 } }

    fn unification_table<'v>(infcx: &'v InferCtxt) -> &'v RefCell<UnificationTable<ty::IntVid>> {
        return &infcx.int_unification_table;
    }

    fn tag(_: Option<ty::IntVid>) -> &'static str {
        "IntVid"
    }
}

impl<'tcx> SimplyUnifiable<'tcx> for IntVarValue {
    fn to_type(&self, tcx: &ty::ctxt<'tcx>) -> Ty<'tcx> {
        match *self {
            ty::IntType(i) => ty::mk_mach_int(tcx, i),
            ty::UintType(i) => ty::mk_mach_uint(tcx, i),
        }
    }

    fn to_type_err(err: expected_found<IntVarValue>) -> ty::type_err<'tcx> {
        return ty::terr_int_mismatch(err);
    }
}

impl UnifyValue for Option<IntVarValue> { }

// Floating point type keys

impl UnifyKey for ty::FloatVid {
    type Value = Option<ast::FloatTy>;

    fn index(&self) -> uint { self.index as uint }

    fn from_index(i: uint) -> ty::FloatVid { ty::FloatVid { index: i as u32 } }

    fn unification_table<'v>(infcx: &'v InferCtxt) -> &'v RefCell<UnificationTable<ty::FloatVid>> {
        return &infcx.float_unification_table;
    }

    fn tag(_: Option<ty::FloatVid>) -> &'static str {
        "FloatVid"
    }
}

impl UnifyValue for Option<ast::FloatTy> {
}

impl<'tcx> SimplyUnifiable<'tcx> for ast::FloatTy {
    fn to_type(&self, tcx: &ty::ctxt<'tcx>) -> Ty<'tcx> {
        ty::mk_mach_float(tcx, *self)
    }

    fn to_type_err(err: expected_found<ast::FloatTy>) -> ty::type_err<'tcx> {
        ty::terr_float_mismatch(err)
    }
}
