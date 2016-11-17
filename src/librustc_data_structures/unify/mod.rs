// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::marker;
use std::fmt::Debug;
use std::marker::PhantomData;
use snapshot_vec as sv;

#[cfg(test)]
mod tests;

/// This trait is implemented by any type that can serve as a type
/// variable. We call such variables *unification keys*. For example,
/// this trait is implemented by `IntVid`, which represents integral
/// variables.
///
/// Each key type has an associated value type `V`. For example, for
/// `IntVid`, this is `Option<IntVarValue>`, representing some
/// (possibly not yet known) sort of integer.
///
/// Clients are expected to provide implementations of this trait; you
/// can see some examples in the `test` module.
pub trait UnifyKey: Copy + Clone + Debug + PartialEq {
    type Value: Clone + PartialEq + Debug;

    fn index(&self) -> u32;

    fn from_index(u: u32) -> Self;

    fn tag(k: Option<Self>) -> &'static str;
}

/// This trait is implemented for unify values that can be
/// combined. This relation should be a monoid.
pub trait Combine {
    fn combine(&self, other: &Self) -> Self;
}

impl Combine for () {
    fn combine(&self, _other: &()) {}
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
pub struct VarValue<K: UnifyKey> {
    parent: K, // if equal to self, this is a root
    value: K::Value, // value assigned (only relevant to root)
    rank: u32, // max depth (only relevant to root)
}

/// Table of unification keys and their values.
pub struct UnificationTable<K: UnifyKey> {
    /// Indicates the current value of each key.
    values: sv::SnapshotVec<Delegate<K>>,
}

/// At any time, users may snapshot a unification table.  The changes
/// made during the snapshot may either be *committed* or *rolled back*.
pub struct Snapshot<K: UnifyKey> {
    // Link snapshot to the key type `K` of the table.
    marker: marker::PhantomData<K>,
    snapshot: sv::Snapshot,
}

#[derive(Copy, Clone)]
struct Delegate<K>(PhantomData<K>);

impl<K: UnifyKey> VarValue<K> {
    fn new_var(key: K, value: K::Value) -> VarValue<K> {
        VarValue::new(key, value, 0)
    }

    fn new(parent: K, value: K::Value, rank: u32) -> VarValue<K> {
        VarValue {
            parent: parent, // this is a root
            value: value,
            rank: rank,
        }
    }

    fn redirect(self, to: K) -> VarValue<K> {
        VarValue { parent: to, ..self }
    }

    fn root(self, rank: u32, value: K::Value) -> VarValue<K> {
        VarValue {
            rank: rank,
            value: value,
            ..self
        }
    }

    /// Returns the key of this node. Only valid if this is a root
    /// node, which you yourself must ensure.
    fn key(&self) -> K {
        self.parent
    }

    fn parent(&self, self_key: K) -> Option<K> {
        self.if_not_self(self.parent, self_key)
    }

    fn if_not_self(&self, key: K, self_key: K) -> Option<K> {
        if key == self_key { None } else { Some(key) }
    }
}

// We can't use V:LatticeValue, much as I would like to,
// because frequently the pattern is that V=Option<U> for some
// other type parameter U, and we have no way to say
// Option<U>:LatticeValue.

impl<K: UnifyKey> UnificationTable<K> {
    pub fn new() -> UnificationTable<K> {
        UnificationTable { values: sv::SnapshotVec::new() }
    }

    /// Starts a new snapshot. Each snapshot must be either
    /// rolled back or committed in a "LIFO" (stack) order.
    pub fn snapshot(&mut self) -> Snapshot<K> {
        Snapshot {
            marker: marker::PhantomData::<K>,
            snapshot: self.values.start_snapshot(),
        }
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
        let len = self.values.len();
        let key: K = UnifyKey::from_index(len as u32);
        self.values.push(VarValue::new_var(key, value));
        debug!("{}: created new key: {:?}", UnifyKey::tag(None::<K>), key);
        key
    }

    /// Find the root node for `vid`. This uses the standard
    /// union-find algorithm with path compression:
    /// <http://en.wikipedia.org/wiki/Disjoint-set_data_structure>.
    ///
    /// NB. This is a building-block operation and you would probably
    /// prefer to call `probe` below.
    fn get(&mut self, vid: K) -> VarValue<K> {
        let index = vid.index() as usize;
        let mut value: VarValue<K> = self.values.get(index).clone();
        match value.parent(vid) {
            Some(redirect) => {
                let root: VarValue<K> = self.get(redirect);
                if root.key() != redirect {
                    // Path compression
                    value.parent = root.key();
                    self.values.set(index, value);
                }
                root
            }
            None => value,
        }
    }

    fn is_root(&self, key: K) -> bool {
        let index = key.index() as usize;
        self.values.get(index).parent(key).is_none()
    }

    /// Sets the value for `vid` to `new_value`. `vid` MUST be a root
    /// node! This is an internal operation used to impl other things.
    fn set(&mut self, key: K, new_value: VarValue<K>) {
        assert!(self.is_root(key));

        debug!("Updating variable {:?} to {:?}", key, new_value);

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
    fn unify(&mut self, root_a: VarValue<K>, root_b: VarValue<K>, new_value: K::Value) -> K {
        debug!("unify(root_a(id={:?}, rank={:?}), root_b(id={:?}, rank={:?}))",
               root_a.key(),
               root_a.rank,
               root_b.key(),
               root_b.rank);

        if root_a.rank > root_b.rank {
            // a has greater rank, so a should become b's parent,
            // i.e., b should redirect to a.
            self.redirect_root(root_a.rank, root_b, root_a, new_value)
        } else if root_a.rank < root_b.rank {
            // b has greater rank, so a should redirect to b.
            self.redirect_root(root_b.rank, root_a, root_b, new_value)
        } else {
            // If equal, redirect one to the other and increment the
            // other's rank.
            self.redirect_root(root_a.rank + 1, root_a, root_b, new_value)
        }
    }

    fn redirect_root(&mut self,
                     new_rank: u32,
                     old_root: VarValue<K>,
                     new_root: VarValue<K>,
                     new_value: K::Value)
                     -> K {
        let old_root_key = old_root.key();
        let new_root_key = new_root.key();
        self.set(old_root_key, old_root.redirect(new_root_key));
        self.set(new_root_key, new_root.root(new_rank, new_value));
        new_root_key
    }
}

impl<K: UnifyKey> sv::SnapshotVecDelegate for Delegate<K> {
    type Value = VarValue<K>;
    type Undo = ();

    fn reverse(_: &mut Vec<VarValue<K>>, _: ()) {}
}

// # Base union-find algorithm, where we are just making sets

impl<'tcx, K: UnifyKey> UnificationTable<K>
    where K::Value: Combine
{
    pub fn union(&mut self, a_id: K, b_id: K) -> K {
        let node_a = self.get(a_id);
        let node_b = self.get(b_id);
        let a_id = node_a.key();
        let b_id = node_b.key();
        if a_id != b_id {
            let new_value = node_a.value.combine(&node_b.value);
            self.unify(node_a, node_b, new_value)
        } else {
            a_id
        }
    }

    pub fn find(&mut self, id: K) -> K {
        self.get(id).key()
    }

    pub fn find_value(&mut self, id: K) -> K::Value {
        self.get(id).value
    }

    pub fn unioned(&mut self, a_id: K, b_id: K) -> bool {
        self.find(a_id) == self.find(b_id)
    }
}

// # Non-subtyping unification
//
// Code to handle keys which carry a value, like ints,
// floats---anything that doesn't have a subtyping relationship we
// need to worry about.

impl<'tcx, K, V> UnificationTable<K>
    where K: UnifyKey<Value = Option<V>>,
          V: Clone + PartialEq + Debug
{
    pub fn unify_var_var(&mut self, a_id: K, b_id: K) -> Result<K, (V, V)> {
        let node_a = self.get(a_id);
        let node_b = self.get(b_id);
        let a_id = node_a.key();
        let b_id = node_b.key();

        if a_id == b_id {
            return Ok(a_id);
        }

        let combined = {
            match (&node_a.value, &node_b.value) {
                (&None, &None) => None,
                (&Some(ref v), &None) |
                (&None, &Some(ref v)) => Some(v.clone()),
                (&Some(ref v1), &Some(ref v2)) => {
                    if *v1 != *v2 {
                        return Err((v1.clone(), v2.clone()));
                    }
                    Some(v1.clone())
                }
            }
        };

        Ok(self.unify(node_a, node_b, combined))
    }

    /// Sets the value of the key `a_id` to `b`. Because simple keys do not have any subtyping
    /// relationships, if `a_id` already has a value, it must be the same as `b`.
    pub fn unify_var_value(&mut self, a_id: K, b: V) -> Result<(), (V, V)> {
        let mut node_a = self.get(a_id);

        match node_a.value {
            None => {
                node_a.value = Some(b);
                self.set(node_a.key(), node_a);
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
        self.get(a_id).value
    }

    pub fn unsolved_variables(&mut self) -> Vec<K> {
        self.values
            .iter()
            .filter_map(|vv| {
                if vv.value.is_some() {
                    None
                } else {
                    Some(vv.key())
                }
            })
            .collect()
    }
}
