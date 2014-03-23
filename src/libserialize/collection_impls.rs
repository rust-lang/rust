// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementations of serialization for structures found in libcollections

use std::uint;
use std::default::Default;
use std::hash::{Hash, Hasher};

use {Decodable, Encodable, Decoder, Encoder};
use collections::{DList, RingBuf, TreeMap, TreeSet, Deque, HashMap, HashSet,
                  TrieMap, TrieSet};
use collections::enum_set::{EnumSet, CLike};

impl<
    S: Encoder,
    T: Encodable<S>
> Encodable<S> for DList<T> {
    fn encode(&self, s: &mut S) {
        s.emit_seq(self.len(), |s| {
            for (i, e) in self.iter().enumerate() {
                s.emit_seq_elt(i, |s| e.encode(s));
            }
        })
    }
}

impl<D:Decoder,T:Decodable<D>> Decodable<D> for DList<T> {
    fn decode(d: &mut D) -> DList<T> {
        let mut list = DList::new();
        d.read_seq(|d, len| {
            for i in range(0u, len) {
                list.push_back(d.read_seq_elt(i, |d| Decodable::decode(d)));
            }
        });
        list
    }
}

impl<
    S: Encoder,
    T: Encodable<S>
> Encodable<S> for RingBuf<T> {
    fn encode(&self, s: &mut S) {
        s.emit_seq(self.len(), |s| {
            for (i, e) in self.iter().enumerate() {
                s.emit_seq_elt(i, |s| e.encode(s));
            }
        })
    }
}

impl<D:Decoder,T:Decodable<D>> Decodable<D> for RingBuf<T> {
    fn decode(d: &mut D) -> RingBuf<T> {
        let mut deque = RingBuf::new();
        d.read_seq(|d, len| {
            for i in range(0u, len) {
                deque.push_back(d.read_seq_elt(i, |d| Decodable::decode(d)));
            }
        });
        deque
    }
}

impl<
    E: Encoder,
    K: Encodable<E> + Eq + TotalOrd,
    V: Encodable<E> + Eq
> Encodable<E> for TreeMap<K, V> {
    fn encode(&self, e: &mut E) {
        e.emit_map(self.len(), |e| {
            let mut i = 0;
            for (key, val) in self.iter() {
                e.emit_map_elt_key(i, |e| key.encode(e));
                e.emit_map_elt_val(i, |e| val.encode(e));
                i += 1;
            }
        })
    }
}

impl<
    D: Decoder,
    K: Decodable<D> + Eq + TotalOrd,
    V: Decodable<D> + Eq
> Decodable<D> for TreeMap<K, V> {
    fn decode(d: &mut D) -> TreeMap<K, V> {
        d.read_map(|d, len| {
            let mut map = TreeMap::new();
            for i in range(0u, len) {
                let key = d.read_map_elt_key(i, |d| Decodable::decode(d));
                let val = d.read_map_elt_val(i, |d| Decodable::decode(d));
                map.insert(key, val);
            }
            map
        })
    }
}

impl<
    S: Encoder,
    T: Encodable<S> + Eq + TotalOrd
> Encodable<S> for TreeSet<T> {
    fn encode(&self, s: &mut S) {
        s.emit_seq(self.len(), |s| {
            let mut i = 0;
            for e in self.iter() {
                s.emit_seq_elt(i, |s| e.encode(s));
                i += 1;
            }
        })
    }
}

impl<
    D: Decoder,
    T: Decodable<D> + Eq + TotalOrd
> Decodable<D> for TreeSet<T> {
    fn decode(d: &mut D) -> TreeSet<T> {
        d.read_seq(|d, len| {
            let mut set = TreeSet::new();
            for i in range(0u, len) {
                set.insert(d.read_seq_elt(i, |d| Decodable::decode(d)));
            }
            set
        })
    }
}

impl<
    S: Encoder,
    T: Encodable<S> + CLike
> Encodable<S> for EnumSet<T> {
    fn encode(&self, s: &mut S) {
        let mut bits = 0;
        for item in self.iter() {
            bits |= item.to_uint();
        }
        s.emit_uint(bits);
    }
}

impl<
    D: Decoder,
    T: Decodable<D> + CLike
> Decodable<D> for EnumSet<T> {
    fn decode(d: &mut D) -> EnumSet<T> {
        let bits = d.read_uint();
        let mut set = EnumSet::empty();
        for bit in range(0, uint::BITS) {
            if bits & (1 << bit) != 0 {
                set.add(CLike::from_uint(1 << bit));
            }
        }
        set
    }
}

impl<
    E: Encoder,
    K: Encodable<E> + Hash<S> + TotalEq,
    V: Encodable<E>,
    S,
    H: Hasher<S>
> Encodable<E> for HashMap<K, V, H> {
    fn encode(&self, e: &mut E) {
        e.emit_map(self.len(), |e| {
            let mut i = 0;
            for (key, val) in self.iter() {
                e.emit_map_elt_key(i, |e| key.encode(e));
                e.emit_map_elt_val(i, |e| val.encode(e));
                i += 1;
            }
        })
    }
}

impl<
    D: Decoder,
    K: Decodable<D> + Hash<S> + TotalEq,
    V: Decodable<D>,
    S,
    H: Hasher<S> + Default
> Decodable<D> for HashMap<K, V, H> {
    fn decode(d: &mut D) -> HashMap<K, V, H> {
        d.read_map(|d, len| {
            let hasher = Default::default();
            let mut map = HashMap::with_capacity_and_hasher(len, hasher);
            for i in range(0u, len) {
                let key = d.read_map_elt_key(i, |d| Decodable::decode(d));
                let val = d.read_map_elt_val(i, |d| Decodable::decode(d));
                map.insert(key, val);
            }
            map
        })
    }
}

impl<
    E: Encoder,
    T: Encodable<E> + Hash<S> + TotalEq,
    S,
    H: Hasher<S>
> Encodable<E> for HashSet<T, H> {
    fn encode(&self, s: &mut E) {
        s.emit_seq(self.len(), |s| {
            let mut i = 0;
            for e in self.iter() {
                s.emit_seq_elt(i, |s| e.encode(s));
                i += 1;
            }
        })
    }
}

impl<
    D: Decoder,
    T: Decodable<D> + Hash<S> + TotalEq,
    S,
    H: Hasher<S> + Default
> Decodable<D> for HashSet<T, H> {
    fn decode(d: &mut D) -> HashSet<T, H> {
        d.read_seq(|d, len| {
            let mut set = HashSet::with_capacity_and_hasher(len, Default::default());
            for i in range(0u, len) {
                set.insert(d.read_seq_elt(i, |d| Decodable::decode(d)));
            }
            set
        })
    }
}

impl<
    E: Encoder,
    V: Encodable<E>
> Encodable<E> for TrieMap<V> {
    fn encode(&self, e: &mut E) {
        e.emit_map(self.len(), |e| {
                for (i, (key, val)) in self.iter().enumerate() {
                    e.emit_map_elt_key(i, |e| key.encode(e));
                    e.emit_map_elt_val(i, |e| val.encode(e));
                }
            });
    }
}

impl<
    D: Decoder,
    V: Decodable<D>
> Decodable<D> for TrieMap<V> {
    fn decode(d: &mut D) -> TrieMap<V> {
        d.read_map(|d, len| {
            let mut map = TrieMap::new();
            for i in range(0u, len) {
                let key = d.read_map_elt_key(i, |d| Decodable::decode(d));
                let val = d.read_map_elt_val(i, |d| Decodable::decode(d));
                map.insert(key, val);
            }
            map
        })
    }
}

impl<S: Encoder> Encodable<S> for TrieSet {
    fn encode(&self, s: &mut S) {
        s.emit_seq(self.len(), |s| {
                for (i, e) in self.iter().enumerate() {
                    s.emit_seq_elt(i, |s| e.encode(s));
                }
            })
    }
}

impl<D: Decoder> Decodable<D> for TrieSet {
    fn decode(d: &mut D) -> TrieSet {
        d.read_seq(|d, len| {
            let mut set = TrieSet::new();
            for i in range(0u, len) {
                set.insert(d.read_seq_elt(i, |d| Decodable::decode(d)));
            }
            set
        })
    }
}
