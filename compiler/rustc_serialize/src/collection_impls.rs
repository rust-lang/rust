//! Implementations of serialization for structures found in liballoc

use std::hash::{BuildHasher, Hash};

use crate::{Decodable, Decoder, Encodable, Encoder};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, LinkedList, VecDeque};
use std::rc::Rc;
use std::sync::Arc;

use smallvec::{Array, SmallVec};

impl<S: Encoder, A: Array<Item: Encodable<S>>> Encodable<S> for SmallVec<A> {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        let slice: &[A::Item] = self;
        slice.encode(s)
    }
}

impl<D: Decoder, A: Array<Item: Decodable<D>>> Decodable<D> for SmallVec<A> {
    fn decode(d: &mut D) -> SmallVec<A> {
        d.read_seq(|d, len| (0..len).map(|_| d.read_seq_elt(|d| Decodable::decode(d))).collect())
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for LinkedList<T> {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_seq(self.len(), |s| {
            for (i, e) in self.iter().enumerate() {
                s.emit_seq_elt(i, |s| e.encode(s))?;
            }
            Ok(())
        })
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for LinkedList<T> {
    fn decode(d: &mut D) -> LinkedList<T> {
        d.read_seq(|d, len| (0..len).map(|_| d.read_seq_elt(|d| Decodable::decode(d))).collect())
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for VecDeque<T> {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_seq(self.len(), |s| {
            for (i, e) in self.iter().enumerate() {
                s.emit_seq_elt(i, |s| e.encode(s))?;
            }
            Ok(())
        })
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for VecDeque<T> {
    fn decode(d: &mut D) -> VecDeque<T> {
        d.read_seq(|d, len| (0..len).map(|_| d.read_seq_elt(|d| Decodable::decode(d))).collect())
    }
}

impl<S: Encoder, K, V> Encodable<S> for BTreeMap<K, V>
where
    K: Encodable<S> + PartialEq + Ord,
    V: Encodable<S>,
{
    fn encode(&self, e: &mut S) -> Result<(), S::Error> {
        e.emit_map(self.len(), |e| {
            for (i, (key, val)) in self.iter().enumerate() {
                e.emit_map_elt_key(i, |e| key.encode(e))?;
                e.emit_map_elt_val(|e| val.encode(e))?;
            }
            Ok(())
        })
    }
}

impl<D: Decoder, K, V> Decodable<D> for BTreeMap<K, V>
where
    K: Decodable<D> + PartialEq + Ord,
    V: Decodable<D>,
{
    fn decode(d: &mut D) -> BTreeMap<K, V> {
        d.read_map(|d, len| {
            let mut map = BTreeMap::new();
            for _ in 0..len {
                let key = d.read_map_elt_key(|d| Decodable::decode(d));
                let val = d.read_map_elt_val(|d| Decodable::decode(d));
                map.insert(key, val);
            }
            map
        })
    }
}

impl<S: Encoder, T> Encodable<S> for BTreeSet<T>
where
    T: Encodable<S> + PartialEq + Ord,
{
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_seq(self.len(), |s| {
            for (i, e) in self.iter().enumerate() {
                s.emit_seq_elt(i, |s| e.encode(s))?;
            }
            Ok(())
        })
    }
}

impl<D: Decoder, T> Decodable<D> for BTreeSet<T>
where
    T: Decodable<D> + PartialEq + Ord,
{
    fn decode(d: &mut D) -> BTreeSet<T> {
        d.read_seq(|d, len| {
            let mut set = BTreeSet::new();
            for _ in 0..len {
                set.insert(d.read_seq_elt(|d| Decodable::decode(d)));
            }
            set
        })
    }
}

impl<E: Encoder, K, V, S> Encodable<E> for HashMap<K, V, S>
where
    K: Encodable<E> + Eq,
    V: Encodable<E>,
    S: BuildHasher,
{
    fn encode(&self, e: &mut E) -> Result<(), E::Error> {
        e.emit_map(self.len(), |e| {
            for (i, (key, val)) in self.iter().enumerate() {
                e.emit_map_elt_key(i, |e| key.encode(e))?;
                e.emit_map_elt_val(|e| val.encode(e))?;
            }
            Ok(())
        })
    }
}

impl<D: Decoder, K, V, S> Decodable<D> for HashMap<K, V, S>
where
    K: Decodable<D> + Hash + Eq,
    V: Decodable<D>,
    S: BuildHasher + Default,
{
    fn decode(d: &mut D) -> HashMap<K, V, S> {
        d.read_map(|d, len| {
            let state = Default::default();
            let mut map = HashMap::with_capacity_and_hasher(len, state);
            for _ in 0..len {
                let key = d.read_map_elt_key(|d| Decodable::decode(d));
                let val = d.read_map_elt_val(|d| Decodable::decode(d));
                map.insert(key, val);
            }
            map
        })
    }
}

impl<E: Encoder, T, S> Encodable<E> for HashSet<T, S>
where
    T: Encodable<E> + Eq,
    S: BuildHasher,
{
    fn encode(&self, s: &mut E) -> Result<(), E::Error> {
        s.emit_seq(self.len(), |s| {
            for (i, e) in self.iter().enumerate() {
                s.emit_seq_elt(i, |s| e.encode(s))?;
            }
            Ok(())
        })
    }
}

impl<E: Encoder, T, S> Encodable<E> for &HashSet<T, S>
where
    T: Encodable<E> + Eq,
    S: BuildHasher,
{
    fn encode(&self, s: &mut E) -> Result<(), E::Error> {
        (**self).encode(s)
    }
}

impl<D: Decoder, T, S> Decodable<D> for HashSet<T, S>
where
    T: Decodable<D> + Hash + Eq,
    S: BuildHasher + Default,
{
    fn decode(d: &mut D) -> HashSet<T, S> {
        d.read_seq(|d, len| {
            let state = Default::default();
            let mut set = HashSet::with_capacity_and_hasher(len, state);
            for _ in 0..len {
                set.insert(d.read_seq_elt(|d| Decodable::decode(d)));
            }
            set
        })
    }
}

impl<E: Encoder, K, V, S> Encodable<E> for indexmap::IndexMap<K, V, S>
where
    K: Encodable<E> + Hash + Eq,
    V: Encodable<E>,
    S: BuildHasher,
{
    fn encode(&self, e: &mut E) -> Result<(), E::Error> {
        e.emit_map(self.len(), |e| {
            for (i, (key, val)) in self.iter().enumerate() {
                e.emit_map_elt_key(i, |e| key.encode(e))?;
                e.emit_map_elt_val(|e| val.encode(e))?;
            }
            Ok(())
        })
    }
}

impl<D: Decoder, K, V, S> Decodable<D> for indexmap::IndexMap<K, V, S>
where
    K: Decodable<D> + Hash + Eq,
    V: Decodable<D>,
    S: BuildHasher + Default,
{
    fn decode(d: &mut D) -> indexmap::IndexMap<K, V, S> {
        d.read_map(|d, len| {
            let state = Default::default();
            let mut map = indexmap::IndexMap::with_capacity_and_hasher(len, state);
            for _ in 0..len {
                let key = d.read_map_elt_key(|d| Decodable::decode(d));
                let val = d.read_map_elt_val(|d| Decodable::decode(d));
                map.insert(key, val);
            }
            map
        })
    }
}

impl<E: Encoder, T, S> Encodable<E> for indexmap::IndexSet<T, S>
where
    T: Encodable<E> + Hash + Eq,
    S: BuildHasher,
{
    fn encode(&self, s: &mut E) -> Result<(), E::Error> {
        s.emit_seq(self.len(), |s| {
            for (i, e) in self.iter().enumerate() {
                s.emit_seq_elt(i, |s| e.encode(s))?;
            }
            Ok(())
        })
    }
}

impl<D: Decoder, T, S> Decodable<D> for indexmap::IndexSet<T, S>
where
    T: Decodable<D> + Hash + Eq,
    S: BuildHasher + Default,
{
    fn decode(d: &mut D) -> indexmap::IndexSet<T, S> {
        d.read_seq(|d, len| {
            let state = Default::default();
            let mut set = indexmap::IndexSet::with_capacity_and_hasher(len, state);
            for _ in 0..len {
                set.insert(d.read_seq_elt(|d| Decodable::decode(d)));
            }
            set
        })
    }
}

impl<E: Encoder, T: Encodable<E>> Encodable<E> for Rc<[T]> {
    fn encode(&self, s: &mut E) -> Result<(), E::Error> {
        let slice: &[T] = self;
        slice.encode(s)
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for Rc<[T]> {
    fn decode(d: &mut D) -> Rc<[T]> {
        let vec: Vec<T> = Decodable::decode(d);
        vec.into()
    }
}

impl<E: Encoder, T: Encodable<E>> Encodable<E> for Arc<[T]> {
    fn encode(&self, s: &mut E) -> Result<(), E::Error> {
        let slice: &[T] = self;
        slice.encode(s)
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for Arc<[T]> {
    fn decode(d: &mut D) -> Arc<[T]> {
        let vec: Vec<T> = Decodable::decode(d);
        vec.into()
    }
}
