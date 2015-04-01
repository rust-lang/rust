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

use middle::ty::{IntVarValue};
use middle::ty::{self, Ty};
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

    fn index(&self) -> u32;

    fn from_index(u: u32) -> Self;

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
    Root(K::Value, usize),
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
    pub rank: usize,
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
        let k = UnifyKey::from_index(index as u32);
        debug!("{}: created new key: {:?}",
               UnifyKey::tag(None::<K>),
               k);
        k
    }

    /// Find the root node for `vid`. This uses the standard
    /// union-find algorithm with path compression:
    /// <http://en.wikipedia.org/wiki/Disjoint-set_data_structure>.
    ///
    /// NB. This is a building-block operation and you would probably
    /// prefer to call `probe` below.
    fn get(&mut self, vid: K) -> Node<K> {
        let index = vid.index() as usize;
        let value = (*self.values.get(index)).clone();
        match value {
            Redirect(redirect) => {
                let node: Node<K> = self.get(redirect.clone());
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
        let index = key.index() as usize;
        match *self.values.get(index) {
            Redirect(..) => false,
            Root(..) => true,
        }
    }

    /// Sets the value for `vid` to `new_value`. `vid` MUST be a root
    /// node! This is an internal operation used to impl other things.
    fn set(&mut self, key: K, new_value: VarValue<K>) {
        assert!(self.is_root(&key));

        debug!("Updating variable {:?} to {:?}",
               key, new_value);

        let index = key.index() as usize;
        self.values.set(index, new_value);
    }

    /// Either redirects `node_a` to `node_b` or vice versa, depending
    /// on the relative rank. The value associated with the new root
    /// will be `new_value`.
    ///
    /// NB: This is the "union" operation of "union-find". It is
    /// really more of a building block. If the values associated with
    /// your key are non-trivial, you would probably prefer to call
    /// `unify_var_var` below.
    fn unify(&mut self, node_a: &Node<K>, node_b: &Node<K>, new_value: K::Value) {
        debug!("unify(node_a(id={:?}, rank={:?}), node_b(id={:?}, rank={:?}))",
               node_a.key,
               node_a.rank,
               node_b.key,
               node_b.rank);

        let (new_root, new_rank) = if node_a.rank > node_b.rank {
            // a has greater rank, so a should become b's parent,
            // i.e., b should redirect to a.
            self.set(node_b.key.clone(), Redirect(node_a.key.clone()));
            (node_a.key.clone(), node_a.rank)
        } else if node_a.rank < node_b.rank {
            // b has greater rank, so a should redirect to b.
            self.set(node_a.key.clone(), Redirect(node_b.key.clone()));
            (node_b.key.clone(), node_b.rank)
        } else {
            // If equal, redirect one to the other and increment the
            // other's rank.
            assert_eq!(node_a.rank, node_b.rank);
            self.set(node_b.key.clone(), Redirect(node_a.key.clone()));
            (node_a.key.clone(), node_a.rank + 1)
        };

        self.set(new_root, Root(new_value, new_rank));
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
// Code to handle keys which carry a value, like ints,
// floats---anything that doesn't have a subtyping relationship we
// need to worry about.

impl<'tcx,K,V> UnificationTable<K>
    where K: UnifyKey<Value=Option<V>>,
          V: Clone+PartialEq,
          Option<V>: UnifyValue,
{
    pub fn unify_var_var(&mut self,
                         a_id: K,
                         b_id: K)
                         -> Result<(),(V,V)>
    {
        let node_a = self.get(a_id);
        let node_b = self.get(b_id);
        let a_id = node_a.key.clone();
        let b_id = node_b.key.clone();

        if a_id == b_id { return Ok(()); }

        let combined = {
            match (&node_a.value, &node_b.value) {
                (&None, &None) => {
                    None
                }
                (&Some(ref v), &None) | (&None, &Some(ref v)) => {
                    Some(v.clone())
                }
                (&Some(ref v1), &Some(ref v2)) => {
                    if *v1 != *v2 {
                        return Err((v1.clone(), v2.clone()));
                    }
                    Some(v1.clone())
                }
            }
        };

        Ok(self.unify(&node_a, &node_b, combined))
    }

    /// Sets the value of the key `a_id` to `b`. Because simple keys do not have any subtyping
    /// relationships, if `a_id` already has a value, it must be the same as `b`.
    pub fn unify_var_value(&mut self,
                           a_id: K,
                           b: V)
                           -> Result<(),(V,V)>
    {
        let node_a = self.get(a_id);
        let a_id = node_a.key.clone();

        match node_a.value {
            None => {
                self.set(a_id, Root(Some(b), node_a.rank));
                Ok(())
            }

            Some(ref a_t) => {
                if *a_t == b {
                    Ok(())
                } else {
                    Err((a_t.clone(), b))
                }
            }
        }
    }

    pub fn has_value(&mut self, id: K) -> bool {
        self.get(id).value.is_some()
    }

    pub fn probe(&mut self, a_id: K) -> Option<V> {
        self.get(a_id).value.clone()
    }
}

///////////////////////////////////////////////////////////////////////////

// Integral type keys

pub trait ToType<'tcx> {
    fn to_type(&self, tcx: &ty::ctxt<'tcx>) -> Ty<'tcx>;
}

impl UnifyKey for ty::IntVid {
    type Value = Option<IntVarValue>;
    fn index(&self) -> u32 { self.index }
    fn from_index(i: u32) -> ty::IntVid { ty::IntVid { index: i } }
    fn tag(_: Option<ty::IntVid>) -> &'static str { "IntVid" }
}

impl<'tcx> ToType<'tcx> for IntVarValue {
    fn to_type(&self, tcx: &ty::ctxt<'tcx>) -> Ty<'tcx> {
        match *self {
            ty::IntType(i) => ty::mk_mach_int(tcx, i),
            ty::UintType(i) => ty::mk_mach_uint(tcx, i),
        }
    }
}

impl UnifyValue for Option<IntVarValue> { }

// Floating point type keys

impl UnifyKey for ty::FloatVid {
    type Value = Option<ast::FloatTy>;
    fn index(&self) -> u32 { self.index }
    fn from_index(i: u32) -> ty::FloatVid { ty::FloatVid { index: i } }
    fn tag(_: Option<ty::FloatVid>) -> &'static str { "FloatVid" }
}

impl UnifyValue for Option<ast::FloatTy> {
}

impl<'tcx> ToType<'tcx> for ast::FloatTy {
    fn to_type(&self, tcx: &ty::ctxt<'tcx>) -> Ty<'tcx> {
        ty::mk_mach_float(tcx, *self)
    }
}
