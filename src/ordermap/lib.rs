
// #![doc(html_root_url = "https://docs.rs/ordermap/0.3/")]

//! [`OrderMap`] is a hash table where the iteration order of the key-value
//! pairs is independent of the hash values of the keys.
//!
//! [`OrderMap`]: struct.OrderMap.html

mod macros;
mod util;
mod equivalent;
mod mutable_keys;

use std::hash::Hash;
use std::hash::BuildHasher;
use std::hash::Hasher;
use std::collections::hash_map::RandomState;
use std::ops::RangeFull;

use std::cmp::{max, Ordering};
use std::fmt;
use std::mem::{replace};
use std::marker::PhantomData;

use self::util::{third, ptrdistance, enumerate};
pub use self::equivalent::Equivalent;
pub use self::mutable_keys::MutableKeys;

fn hash_elem_using<B: BuildHasher, K: ?Sized + Hash>(build: &B, k: &K) -> HashValue {
    let mut h = build.build_hasher();
    k.hash(&mut h);
    HashValue(h.finish() as usize)
}

/// Hash value newtype. Not larger than usize, since anything larger
/// isn't used for selecting position anyway.
#[derive(Copy, Debug)]
struct HashValue(usize);

impl Clone for HashValue {
    #[inline]
    fn clone(&self) -> Self { *self }
}
impl PartialEq for HashValue {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.0 == rhs.0
    }
}

/// A possibly truncated hash value.
///
#[derive(Debug)]
struct ShortHash<Sz>(usize, PhantomData<Sz>);

impl<Sz> ShortHash<Sz> {
    /// Pretend this is a full HashValue, which
    /// is completely ok w.r.t determining bucket index
    ///
    /// - Sz = u32: 32-bit hash is enough to select bucket index
    /// - Sz = u64: hash is not truncated
    fn into_hash(self) -> HashValue {
        HashValue(self.0)
    }
}

impl<Sz> Copy for ShortHash<Sz> { }
impl<Sz> Clone for ShortHash<Sz> {
    #[inline]
    fn clone(&self) -> Self { *self }
}

impl<Sz> PartialEq for ShortHash<Sz> {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.0 == rhs.0
    }
}

// Compare ShortHash == HashValue by truncating appropriately
// if applicable before the comparison
impl<Sz> PartialEq<HashValue> for ShortHash<Sz> where Sz: Size {
    #[inline]
    fn eq(&self, rhs: &HashValue) -> bool {
        if Sz::is_64_bit() {
            self.0 == rhs.0
        } else {
            lo32(self.0 as u64) == lo32(rhs.0 as u64)
        }
    }
}
impl<Sz> From<ShortHash<Sz>> for HashValue {
    fn from(x: ShortHash<Sz>) -> Self { HashValue(x.0) }
}

/// `Pos` is stored in the `indices` array and it points to the index of a
/// `Bucket` in self.entries.
///
/// Pos can be interpreted either as a 64-bit index, or as a 32-bit index and
/// a 32-bit hash.
///
/// Storing the truncated hash next to the index saves loading the hash from the
/// entry, increasing the cache efficiency.
///
/// Note that the lower 32 bits of the hash is enough to compute desired
/// position and probe distance in a hash map with less than 2**32 buckets.
///
/// The OrderMap will simply query its **current raw capacity** to see what its
/// current size class is, and dispatch to the 32-bit or 64-bit lookup code as
/// appropriate. Only the growth code needs some extra logic to handle the
/// transition from one class to another
#[derive(Copy)]
struct Pos {
    index: u64,
}

impl Clone for Pos {
    #[inline(always)]
    fn clone(&self) -> Self { *self }
}

impl fmt::Debug for Pos {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.pos() {
            Some(i) => write!(f, "Pos({} / {:x})", i, self.index),
            None => write!(f, "Pos(None)"),
        }
    }
}

impl Pos {
    #[inline]
    fn none() -> Self { Pos { index: !0 } }

    #[inline]
    fn is_none(&self) -> bool { self.index == !0 }

    #[inline]
    fn pos(&self) -> Option<usize> {
        if self.index == !0 { None } else { Some(lo32(self.index as u64)) }
    }

    #[inline]
    fn with_hash<Sz>(i: usize, hash: HashValue) -> Self
        where Sz: Size
    {
        if Sz::is_64_bit() {
            Pos {
                index: i as u64,
            }
        } else {
            Pos {
                index: i as u64 | ((hash.0 as u64) << 32)
            }
        }
    }

    #[inline]
    fn resolve<Sz>(&self) -> Option<(usize, ShortHashProxy<Sz>)>
        where Sz: Size
    {
        if Sz::is_64_bit() {
            if !self.is_none() {
                Some((self.index as usize, ShortHashProxy::new(0)))
            } else {
                None
            }
        } else {
            if !self.is_none() {
                let (i, hash) = split_lo_hi(self.index);
                Some((i as usize, ShortHashProxy::new(hash as usize)))
            } else {
                None
            }
        }
    }
}

#[inline]
fn lo32(x: u64) -> usize { (x & 0xFFFF_FFFF) as usize }

// split into low, hi parts
#[inline]
fn split_lo_hi(x: u64) -> (u32, u32) { (x as u32, (x >> 32) as u32) }

// Possibly contains the truncated hash value for an entry, depending on
// the size class.
struct ShortHashProxy<Sz>(usize, PhantomData<Sz>);

impl<Sz> ShortHashProxy<Sz>
    where Sz: Size
{
    fn new(x: usize) -> Self {
        ShortHashProxy(x, PhantomData)
    }

    /// Get the hash from either `self` or from a lookup into `entries`,
    /// depending on `Sz`.
    fn get_short_hash<K, V>(&self, entries: &[Bucket<K, V>], index: usize) -> ShortHash<Sz> {
        if Sz::is_64_bit() {
            ShortHash(entries[index].hash.0, PhantomData)
        } else {
            ShortHash(self.0, PhantomData)
        }
    }
}

/// A hash table where the iteration order of the key-value pairs is independent
/// of the hash values of the keys.
///
/// The interface is closely compatible with the standard `HashMap`, but also
/// has additional features.
///
/// # Order
///
/// The key-value pairs have a consistent order that is determined by
/// the sequence of insertion and removal calls on the map. The order does
/// not depend on the keys or the hash function at all.
///
/// All iterators traverse the map in *the order*.
///
/// # Indices
///
/// The key-value pairs are indexed in a compact range without holes in the
/// range `0..self.len()`. For example, the method `.get_full` looks up the
/// index for a key, and the method `.get_index` looks up the key-value pair by
/// index.
///
/// # Examples
///
/// ```
/// use ordermap::OrderMap;
///
/// // count the frequency of each letter in a sentence.
/// let mut letters = OrderMap::new();
/// for ch in "a short treatise on fungi".chars() {
///     *letters.entry(ch).or_insert(0) += 1;
/// }
///
/// assert_eq!(letters[&'s'], 2);
/// assert_eq!(letters[&'t'], 3);
/// assert_eq!(letters[&'u'], 1);
/// assert_eq!(letters.get(&'y'), None);
/// ```
#[derive(Clone)]
pub struct OrderMap<K, V, S = RandomState> {
    mask: usize,
    /// indices are the buckets. indices.len() == raw capacity
    indices: Box<[Pos]>,
    /// entries is a dense vec of entries in their order. entries.len() == len
    entries: Vec<Bucket<K, V>>,
    hash_builder: S,
}

#[derive(Copy, Clone, Debug)]
struct Bucket<K, V> {
    hash: HashValue,
    key: K,
    value: V,
}

#[inline(always)]
fn desired_pos(mask: usize, hash: HashValue) -> usize {
    hash.0 & mask
}

/// The number of steps that `current` is forward of the desired position for hash
#[inline(always)]
fn probe_distance(mask: usize, hash: HashValue, current: usize) -> usize {
    current.wrapping_sub(desired_pos(mask, hash)) & mask
}

enum Inserted<V> {
    Done,
    Swapped { prev_value: V },
    RobinHood {
        probe: usize,
        old_pos: Pos,
    }
}

impl<K, V, S> fmt::Debug for OrderMap<K, V, S>
    where K: fmt::Debug + Hash + Eq,
          V: fmt::Debug,
          S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(f.debug_map().entries(self.iter()).finish());
        try!(writeln!(f, ""));
        for (i, index) in enumerate(&*self.indices) {
            try!(write!(f, "{}: {:?}", i, index));
            if let Some(pos) = index.pos() {
                let hash = self.entries[pos].hash;
                let key = &self.entries[pos].key;
                let desire = desired_pos(self.mask, hash);
                try!(writeln!(f, ", desired={}, probe_distance={}, key={:?}",
                              desire,
                              probe_distance(self.mask, hash, i),
                              key));
            }
            try!(writeln!(f, ""));
        }
        try!(writeln!(f, "cap={}, raw_cap={}, entries.cap={}",
                      self.capacity(),
                      self.raw_capacity(),
                      self.entries.capacity()));
        Ok(())
    }
}

#[inline]
fn usable_capacity(cap: usize) -> usize {
    cap - cap / 4
}

#[inline]
fn to_raw_capacity(n: usize) -> usize {
    n + n / 3
}

// this could not be captured in an efficient iterator
macro_rules! probe_loop {
    ($probe_var: ident < $len: expr, $body: expr) => {
        loop {
            if $probe_var < $len {
                $body
                $probe_var += 1;
            } else {
                $probe_var = 0;
            }
        }
    }
}

impl<K, V> OrderMap<K, V> {
    /// Create a new map. (Does not allocate.)
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Create a new map with capacity for `n` key-value pairs. (Does not
    /// allocate if `n` is zero.)
    ///
    /// Computes in **O(n)** time.
    pub fn with_capacity(n: usize) -> Self {
        Self::with_capacity_and_hasher(n, <_>::default())
    }
}

impl<K, V, S> OrderMap<K, V, S>
{
    /// Create a new map with capacity for `n` key-value pairs. (Does not
    /// allocate if `n` is zero.)
    ///
    /// Computes in **O(n)** time.
    pub fn with_capacity_and_hasher(n: usize, hash_builder: S) -> Self
        where S: BuildHasher
    {
        if n == 0 {
            OrderMap {
                mask: 0,
                indices: Box::new([]),
                entries: Vec::new(),
                hash_builder: hash_builder,
            }
        } else {
            let raw = to_raw_capacity(n);
            let raw_cap = max(raw.next_power_of_two(), 8);
            OrderMap {
                mask: raw_cap.wrapping_sub(1),
                indices: vec![Pos::none(); raw_cap].into_boxed_slice(),
                entries: Vec::with_capacity(usable_capacity(raw_cap)),
                hash_builder: hash_builder,
            }
        }
    }

    /// Return the number of key-value pairs in the map.
    ///
    /// Computes in **O(1)** time.
    pub fn len(&self) -> usize { self.entries.len() }

    /// Returns true if the map contains no elements.
    ///
    /// Computes in **O(1)** time.
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Create a new map with `hash_builder`
    pub fn with_hasher(hash_builder: S) -> Self
        where S: BuildHasher
    {
        Self::with_capacity_and_hasher(0, hash_builder)
    }

    /// Return a reference to the map's `BuildHasher`.
    pub fn hasher(&self) -> &S
        where S: BuildHasher
    {
        &self.hash_builder
    }

    // Return whether we need 32 or 64 bits to specify a bucket or entry index
    // #[cfg(not(feature = "test_low_transition_point"))]
    fn size_class_is_64bit(&self) -> bool {
        usize::max_value() > u32::max_value() as usize &&
            self.raw_capacity() >= u32::max_value() as usize
    }

    // for testing
    // #[cfg(feature = "test_low_transition_point")]
    // fn size_class_is_64bit(&self) -> bool {
    //     self.raw_capacity() >= 64
    // }

    #[inline(always)]
    fn raw_capacity(&self) -> usize {
        self.indices.len()
    }

    /// Computes in **O(1)** time.
    pub fn capacity(&self) -> usize {
        usable_capacity(self.raw_capacity())
    }
}

/// Trait for the "size class". Either u32 or u64 depending on the index
/// size needed to address an entry's indes in self.entries.
trait Size {
    fn is_64_bit() -> bool;
    fn is_same_size<T: Size>() -> bool {
        Self::is_64_bit() == T::is_64_bit()
    }
}

impl Size for u32 {
    #[inline]
    fn is_64_bit() -> bool { false }
}

impl Size for u64 {
    #[inline]
    fn is_64_bit() -> bool { true }
}

/// Call self.method(args) with `::<u32>` or `::<u64>` depending on `self`
/// size class.
///
/// The u32 or u64 is *prepended* to the type parameter list!
macro_rules! dispatch_32_vs_64 {
    ($self_:ident . $method:ident::<$($t:ty),*>($($arg:expr),*)) => {
        if $self_.size_class_is_64bit() {
            $self_.$method::<u64, $($t),*>($($arg),*)
        } else {
            $self_.$method::<u32, $($t),*>($($arg),*)
        }
    };
    ($self_:ident . $method:ident ($($arg:expr),*)) => {
        if $self_.size_class_is_64bit() {
            $self_.$method::<u64>($($arg),*)
        } else {
            $self_.$method::<u32>($($arg),*)
        }
    };
}

/// FIXME: Remove dependence on the `S` parameter
/// (to match HashMap).
pub enum Entry<'a, K: 'a, V: 'a, S: 'a = RandomState> {
    Occupied(OccupiedEntry<'a, K, V, S>),
    Vacant(VacantEntry<'a, K, V, S>),
}

impl<'a, K, V, S> Entry<'a, K, V, S> {
    /// Computes in **O(1)** time (amortized average).
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default),
        }
    }

    /// Computes in **O(1)** time (amortized average).
    pub fn or_insert_with<F>(self, call: F) -> &'a mut V
        where F: FnOnce() -> V,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(call()),
        }
    }

    pub fn key(&self) -> &K {
        match *self {
            Entry::Occupied(ref entry) => entry.key(),
            Entry::Vacant(ref entry) => entry.key(),
        }
    }

    pub fn index(&self) -> usize {
        match *self {
            Entry::Occupied(ref entry) => entry.index(),
            Entry::Vacant(ref entry) => entry.index(),
        }
    }
}

pub struct OccupiedEntry<'a, K: 'a, V: 'a, S: 'a = RandomState> {
    map: &'a mut OrderMap<K, V, S>,
    key: K,
    probe: usize,
    index: usize,
}

impl<'a, K, V, S> OccupiedEntry<'a, K, V, S> {
    pub fn key(&self) -> &K { &self.key }
    pub fn get(&self) -> &V {
        &self.map.entries[self.index].value
    }
    pub fn get_mut(&mut self) -> &mut V {
        &mut self.map.entries[self.index].value
    }

    pub fn index(&self) -> usize {
        self.index
    }
    pub fn into_mut(self) -> &'a mut V {
        &mut self.map.entries[self.index].value
    }

    pub fn insert(self, value: V) -> V {
        replace(&mut self.into_mut(), value)
    }

    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    /// Remove and return the key, value pair stored in the map for this entry
    pub fn remove_entry(self) -> (K, V) {
        self.map.remove_found(self.probe, self.index)
    }
}


pub struct VacantEntry<'a, K: 'a, V: 'a, S: 'a = RandomState> {
    map: &'a mut OrderMap<K, V, S>,
    key: K,
    hash: HashValue,
    probe: usize,
}

impl<'a, K, V, S> VacantEntry<'a, K, V, S> {
    pub fn key(&self) -> &K { &self.key }
    pub fn into_key(self) -> K { self.key }
    pub fn index(&self) -> usize { self.map.len() }
    pub fn insert(self, value: V) -> &'a mut V {
        if self.map.size_class_is_64bit() {
            self.insert_impl::<u64>(value)
        } else {
            self.insert_impl::<u32>(value)
        }
    }

    fn insert_impl<Sz>(self, value: V) -> &'a mut V
        where Sz: Size
    {
        let index = self.map.entries.len();
        self.map.entries.push(Bucket { hash: self.hash, key: self.key, value: value });
        let old_pos = Pos::with_hash::<Sz>(index, self.hash);
        self.map.insert_phase_2::<Sz>(self.probe, old_pos);
        &mut {self.map}.entries[index].value
    }
}

impl<K, V, S> OrderMap<K, V, S>
    where K: Hash + Eq,
          S: BuildHasher,
{
    /// Get the given key’s corresponding entry in the map for in-place manipulation.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn entry(&mut self, key: K) -> Entry<K, V, S> {
        self.reserve_one();
        dispatch_32_vs_64!(self.entry_phase_1(key))
    }

    // Warning, this is a code duplication zone Entry is not yet finished
    fn entry_phase_1<Sz>(&mut self, key: K) -> Entry<K, V, S>
        where Sz: Size
    {
        let hash = hash_elem_using(&self.hash_builder, &key);
        let mut probe = desired_pos(self.mask, hash);
        let mut dist = 0;
        debug_assert!(self.len() < self.raw_capacity());
        probe_loop!(probe < self.indices.len(), {
            if let Some((i, hash_proxy)) = self.indices[probe].resolve::<Sz>() {
                let entry_hash = hash_proxy.get_short_hash(&self.entries, i);
                // if existing element probed less than us, swap
                let their_dist = probe_distance(self.mask, entry_hash.into_hash(), probe);
                if their_dist < dist {
                    // robin hood: steal the spot if it's better for us
                    return Entry::Vacant(VacantEntry {
                        map: self,
                        hash: hash,
                        key: key,
                        probe: probe,
                    });
                } else if entry_hash == hash && self.entries[i].key == key {
                    return Entry::Occupied(OccupiedEntry {
                        map: self,
                        key: key,
                        probe: probe,
                        index: i,
                    });
                }
            } else {
                // empty bucket, insert here
                return Entry::Vacant(VacantEntry {
                    map: self,
                    hash: hash,
                    key: key,
                    probe: probe,
                });
            }
            dist += 1;
        });
    }

    /// Remove all key-value pairs in the map, while preserving its capacity.
    ///
    /// Computes in **O(n)** time.
    pub fn clear(&mut self) {
        self.entries.clear();
        for pos in &mut *self.indices {
            *pos = Pos::none();
        }
    }

    /// FIXME Not implemented fully yet
    pub fn reserve(&mut self, additional: usize) {
        if additional > 0 {
            self.reserve_one();
        }
    }

    // First phase: Look for the preferred location for key.
    //
    // We will know if `key` is already in the map, before we need to insert it.
    // When we insert they key, it might be that we need to continue displacing
    // entries (robin hood hashing), in which case Inserted::RobinHood is returned
    fn insert_phase_1<Sz>(&mut self, key: K, value: V) -> Inserted<V>
        where Sz: Size
    {
        let hash = hash_elem_using(&self.hash_builder, &key);
        let mut probe = desired_pos(self.mask, hash);
        let mut dist = 0;
        let insert_kind;
        debug_assert!(self.len() < self.raw_capacity());
        probe_loop!(probe < self.indices.len(), {
            let pos = &mut self.indices[probe];
            if let Some((i, hash_proxy)) = pos.resolve::<Sz>() {
                let entry_hash = hash_proxy.get_short_hash(&self.entries, i);
                // if existing element probed less than us, swap
                let their_dist = probe_distance(self.mask, entry_hash.into_hash(), probe);
                if their_dist < dist {
                    // robin hood: steal the spot if it's better for us
                    let index = self.entries.len();
                    insert_kind = Inserted::RobinHood {
                        probe: probe,
                        old_pos: Pos::with_hash::<Sz>(index, hash),
                    };
                    break;
                } else if entry_hash == hash && self.entries[i].key == key {
                    return Inserted::Swapped {
                        prev_value: replace(&mut self.entries[i].value, value),
                    };
                }
            } else {
                // empty bucket, insert here
                let index = self.entries.len();
                *pos = Pos::with_hash::<Sz>(index, hash);
                insert_kind = Inserted::Done;
                break;
            }
            dist += 1;
        });
        self.entries.push(Bucket { hash: hash, key: key, value: value });
        insert_kind
    }

    fn first_allocation(&mut self) {
        debug_assert_eq!(self.len(), 0);
        let raw_cap = 8usize;
        self.mask = raw_cap.wrapping_sub(1);
        self.indices = vec![Pos::none(); raw_cap].into_boxed_slice();
        self.entries = Vec::with_capacity(usable_capacity(raw_cap));
    }

    #[inline(never)]
    // `Sz` is *current* Size class, before grow
    fn double_capacity<Sz>(&mut self)
        where Sz: Size
    {
        debug_assert!(self.raw_capacity() == 0 || self.len() > 0);
        if self.raw_capacity() == 0 {
            return self.first_allocation();
        }

        // find first ideally placed element -- start of cluster
        let mut first_ideal = 0;
        for (i, index) in enumerate(&*self.indices) {
            if let Some(pos) = index.pos() {
                if 0 == probe_distance(self.mask, self.entries[pos].hash, i) {
                    first_ideal = i;
                    break;
                }
            }
        }

        // visit the entries in an order where we can simply reinsert them
        // into self.indices without any bucket stealing.
        let new_raw_cap = self.indices.len() * 2;
        let old_indices = replace(&mut self.indices, vec![Pos::none(); new_raw_cap].into_boxed_slice());
        self.mask = new_raw_cap.wrapping_sub(1);

        // `Sz` is the old size class, and either u32 or u64 is the new
        for &pos in &old_indices[first_ideal..] {
            dispatch_32_vs_64!(self.reinsert_entry_in_order::<Sz>(pos));
        }

        for &pos in &old_indices[..first_ideal] {
            dispatch_32_vs_64!(self.reinsert_entry_in_order::<Sz>(pos));
        }
        let more = self.capacity() - self.len();
        self.entries.reserve_exact(more);
    }

    // write to self.indices
    // read from self.entries at `pos`
    //
    // reinserting rewrites all `Pos` entries anyway. This handles transitioning
    // from u32 to u64 size class if needed by using the two type parameters.
    fn reinsert_entry_in_order<SzNew, SzOld>(&mut self, pos: Pos)
        where SzNew: Size,
              SzOld: Size,
    {
        if let Some((i, hash_proxy)) = pos.resolve::<SzOld>() {
            // only if the size class is conserved can we use the short hash
            let entry_hash = if SzOld::is_same_size::<SzNew>() {
                hash_proxy.get_short_hash(&self.entries, i).into_hash()
            } else {
                self.entries[i].hash
            };
            // find first empty bucket and insert there
            let mut probe = desired_pos(self.mask, entry_hash);
            probe_loop!(probe < self.indices.len(), {
                if let Some(_) = self.indices[probe].resolve::<SzNew>() {
                    /* nothing */
                } else {
                    // empty bucket, insert here
                    self.indices[probe] = Pos::with_hash::<SzNew>(i, entry_hash);
                    return;
                }
            });
        }
    }

    fn reserve_one(&mut self) {
        if self.len() == self.capacity() {
            dispatch_32_vs_64!(self.double_capacity());
        }
    }

    /// Insert they key-value pair into the map.
    ///
    /// If a value already existed for `key`, that old value is returned
    /// in `Some`; otherwise, return `None`.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.reserve_one();
        if self.size_class_is_64bit() {
            match self.insert_phase_1::<u64>(key, value) {
                Inserted::Swapped { prev_value } => Some(prev_value),
                Inserted::Done => None,
                Inserted::RobinHood { probe, old_pos } => {
                    self.insert_phase_2::<u64>(probe, old_pos);
                    None
                }
            }
        } else {
            match self.insert_phase_1::<u32>(key, value) {
                Inserted::Swapped { prev_value } => Some(prev_value),
                Inserted::Done => None,
                Inserted::RobinHood { probe, old_pos } => {
                    self.insert_phase_2::<u32>(probe, old_pos);
                    None
                }
            }
        }
    }

    /// Return an iterator over the key-value pairs of the map, in their order
    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            iter: self.entries.iter()
        }
    }

    /// Return an iterator over the key-value pairs of the map, in their order
    pub fn iter_mut(&mut self) -> IterMut<K, V> {
        IterMut {
            iter: self.entries.iter_mut()
        }
    }

    /// Return an iterator over the keys of the map, in their order
    pub fn keys(&self) -> Keys<K, V> {
        Keys {
            iter: self.entries.iter()
        }
    }

    /// Return an iterator over the values of the map, in their order
    pub fn values(&self) -> Values<K, V> {
        Values {
            iter: self.entries.iter()
        }
    }

    /// Return an iterator over mutable references to the the values of the map,
    /// in their order
    pub fn values_mut(&mut self) -> ValuesMut<K, V> {
        ValuesMut {
            iter: self.entries.iter_mut()
        }
    }

    /// Return `true` if and equivalent to `key` exists in the map.
    ///
    /// Computes in **O(1)** time (average).
    pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool
        where Q: Hash + Equivalent<K>,
    {
        self.find(key).is_some()
    }

    /// Return a reference to the value stored for `key`, if it is present,
    /// else `None`.
    ///
    /// Computes in **O(1)** time (average).
    pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<&V>
        where Q: Hash + Equivalent<K>,
    {
        self.get_full(key).map(third)
    }

    /// Return item index, key and value
    pub fn get_full<Q: ?Sized>(&self, key: &Q) -> Option<(usize, &K, &V)>
        where Q: Hash + Equivalent<K>,
    {
        if let Some((_, found)) = self.find(key) {
            let entry = &self.entries[found];
            Some((found, &entry.key, &entry.value))
        } else {
            None
        }
    }

    pub fn get_mut<Q: ?Sized>(&mut self, key: &Q) -> Option<&mut V>
        where Q: Hash + Equivalent<K>,
    {
        self.get_full_mut(key).map(third)
    }

    pub fn get_full_mut<Q: ?Sized>(&mut self, key: &Q)
        -> Option<(usize, &K, &mut V)>
        where Q: Hash + Equivalent<K>,
    {
        self.get_full_mut2(key).map(|(i, k, v)| (i, &*k, v))
    }

    /// Return probe (indices) and position (entries)
    fn find<Q: ?Sized>(&self, key: &Q) -> Option<(usize, usize)>
        where Q: Hash + Equivalent<K>,
    {
        if self.len() == 0 { return None; }
        let h = hash_elem_using(&self.hash_builder, key);
        self.find_using(h, move |entry| { Q::equivalent(key, &entry.key) })
    }

    /// FIXME Same as .swap_remove
    ///
    /// Computes in **O(1)** time (average).
    pub fn remove<Q: ?Sized>(&mut self, key: &Q) -> Option<V>
        where Q: Hash + Equivalent<K>,
    {
        self.swap_remove(key)
    }

    /// Remove the key-value pair equivalent to `key` and return
    /// its value.
    ///
    /// Like `Vec::swap_remove`, the pair is removed by swapping it with the
    /// last element of the map and popping it off. **This perturbs
    /// the postion of what used to be the last element!**
    ///
    /// Return `None` if `key` is not in map.
    ///
    /// Computes in **O(1)** time (average).
    pub fn swap_remove<Q: ?Sized>(&mut self, key: &Q) -> Option<V>
        where Q: Hash + Equivalent<K>,
    {
        self.swap_remove_full(key).map(third)
    }

    /// Remove the key-value pair equivalent to `key` and return it and
    /// the index it had.
    ///
    /// Like `Vec::swap_remove`, the pair is removed by swapping it with the
    /// last element of the map and popping it off. **This perturbs
    /// the postion of what used to be the last element!**
    ///
    /// Return `None` if `key` is not in map.
    pub fn swap_remove_full<Q: ?Sized>(&mut self, key: &Q) -> Option<(usize, K, V)>
        where Q: Hash + Equivalent<K>,
    {
        let (probe, found) = match self.find(key) {
            None => return None,
            Some(t) => t,
        };
        let (k, v) = self.remove_found(probe, found);
        Some((found, k, v))
    }

    /// Remove the last key-value pair
    ///
    /// Computes in **O(1)** time (average).
    pub fn pop(&mut self) -> Option<(K, V)> {
        self.pop_impl()
    }

    /// Scan through each key-value pair in the map and keep those where the
    /// closure `keep` returns `true`.
    ///
    /// The order the elements are visited is not specified.
    ///
    /// Computes in **O(n)** time (average).
    pub fn retain<F>(&mut self, mut keep: F)
        where F: FnMut(&K, &mut V) -> bool,
    {
        self.retain2(move |k, v| keep(&*k, v))
    }

    /// Sort the key-value pairs of the map and return a by value iterator of
    /// the key-value pairs with the result.
    ///
    /// The sort is stable.
    pub fn sorted_by<F>(mut self, mut cmp: F) -> IntoIter<K, V>
        where F: FnMut(&K, &V, &K, &V) -> Ordering
    {
        self.entries.sort_by(move |a, b| cmp(&a.key, &a.value, &b.key, &b.value));
        self.into_iter()
    }
    /// Clears the `OrderMap`, returning all key-value pairs as a drain iterator.
    /// Keeps the allocated memory for reuse.
    pub fn drain(&mut self, range: RangeFull) -> Drain<K, V> {
        for i in &mut *self.indices {
            *i = Pos::none();
        }

        Drain {
            inner: self.entries.drain(range),
        }
    }
}

impl<K, V, S> OrderMap<K, V, S> {
    /// Get a key-value pair by index
    ///
    /// Valid indices are *0 <= index < self.len()*
    ///
    /// Computes in **O(1)** time.
    pub fn get_index(&self, index: usize) -> Option<(&K, &V)> {
        self.entries.get(index).map(|ent| (&ent.key, &ent.value))
    }

    /// Get a key-value pair by index
    ///
    /// Valid indices are *0 <= index < self.len()*
    ///
    /// Computes in **O(1)** time.
    pub fn get_index_mut(&mut self, index: usize) -> Option<(&mut K, &mut V)> {
        self.entries.get_mut(index).map(|ent| (&mut ent.key, &mut ent.value))
    }

    /// Remove the key-value pair by index
    ///
    /// Valid indices are *0 <= index < self.len()*
    ///
    /// Computes in **O(1)** time (average).
    pub fn swap_remove_index(&mut self, index: usize) -> Option<(K, V)> {
        let (probe, found) = match self.entries.get(index)
            .map(|e| self.find_existing_entry(e))
        {
            None => return None,
            Some(t) => t,
        };
        Some(self.remove_found(probe, found))
    }
}

// Methods that don't use any properties (Hash / Eq) of K.
//
// It's cleaner to separate them out, then the compiler checks that we are not
// using Hash + Eq at all in these methods.
//
// However, we should probably not let this show in the public API or docs.
impl<K, V, S> OrderMap<K, V, S> {
    fn pop_impl(&mut self) -> Option<(K, V)> {
        let (probe, found) = match self.entries.last()
            .map(|e| self.find_existing_entry(e))
        {
            None => return None,
            Some(t) => t,
        };
        debug_assert_eq!(found, self.entries.len() - 1);
        Some(self.remove_found(probe, found))
    }

    /// phase 2 is post-insert where we forward-shift `Pos` in the indices.
    fn insert_phase_2<Sz>(&mut self, mut probe: usize, mut old_pos: Pos)
        where Sz: Size
    {
        probe_loop!(probe < self.indices.len(), {
            let pos = &mut self.indices[probe];
            if pos.is_none() {
                *pos = old_pos;
                break;
            } else {
                old_pos = replace(pos, old_pos);
            }
        });
    }


    /// Return probe (indices) and position (entries)
    fn find_using<F>(&self, hash: HashValue, key_eq: F) -> Option<(usize, usize)>
        where F: Fn(&Bucket<K, V>) -> bool,
    {
        dispatch_32_vs_64!(self.find_using_impl::<_>(hash, key_eq))
    }

    fn find_using_impl<Sz, F>(&self, hash: HashValue, key_eq: F) -> Option<(usize, usize)>
        where F: Fn(&Bucket<K, V>) -> bool,
              Sz: Size,
    {
        debug_assert!(self.len() > 0);
        let mut probe = desired_pos(self.mask, hash);
        let mut dist = 0;
        probe_loop!(probe < self.indices.len(), {
            if let Some((i, hash_proxy)) = self.indices[probe].resolve::<Sz>() {
                let entry_hash = hash_proxy.get_short_hash(&self.entries, i);
                if dist > probe_distance(self.mask, entry_hash.into_hash(), probe) {
                    // give up when probe distance is too long
                    return None;
                } else if entry_hash == hash && key_eq(&self.entries[i]) {
                    return Some((probe, i));
                }
            } else {
                return None;
            }
            dist += 1;
        });
    }

    /// Find `entry` which is already placed inside self.entries;
    /// return its probe and entry index.
    fn find_existing_entry(&self, entry: &Bucket<K, V>) -> (usize, usize)
    {
        dispatch_32_vs_64!(self.find_existing_entry_impl(entry))
    }

    fn find_existing_entry_impl<Sz>(&self, entry: &Bucket<K, V>) -> (usize, usize)
        where Sz: Size,
    {
        debug_assert!(self.len() > 0);
        let hash = entry.hash;
        let actual_pos = ptrdistance(&self.entries[0], entry);
        let mut probe = desired_pos(self.mask, hash);
        probe_loop!(probe < self.indices.len(), {
            if let Some((i, _)) = self.indices[probe].resolve::<Sz>() {
                if i == actual_pos {
                    return (probe, actual_pos);
                }
            } else {
                debug_assert!(false, "the entry does not exist");
            }
        });
    }

    fn remove_found(&mut self, probe: usize, found: usize) -> (K, V) {
        dispatch_32_vs_64!(self.remove_found_impl(probe, found))
    }

    fn remove_found_impl<Sz>(&mut self, probe: usize, found: usize) -> (K, V)
        where Sz: Size
    {
        // index `probe` and entry `found` is to be removed
        // use swap_remove, but then we need to update the index that points
        // to the other entry that has to move
        self.indices[probe] = Pos::none();
        let entry = self.entries.swap_remove(found);

        // correct index that points to the entry that had to swap places
        if let Some(entry) = self.entries.get(found) {
            // was not last element
            // examine new element in `found` and find it in indices
            let mut probe = desired_pos(self.mask, entry.hash);
            probe_loop!(probe < self.indices.len(), {
                if let Some((i, _)) = self.indices[probe].resolve::<Sz>() {
                    if i >= self.entries.len() {
                        // found it
                        self.indices[probe] = Pos::with_hash::<Sz>(found, entry.hash);
                        break;
                    }
                }
            });
        }
        // backward shift deletion in self.indices
        // after probe, shift all non-ideally placed indices backward
        if self.len() > 0 {
            let mut last_probe = probe;
            let mut probe = probe + 1;
            probe_loop!(probe < self.indices.len(), {
                if let Some((i, hash_proxy)) = self.indices[probe].resolve::<Sz>() {
                    let entry_hash = hash_proxy.get_short_hash(&self.entries, i);
                    if probe_distance(self.mask, entry_hash.into_hash(), probe) > 0 {
                        self.indices[last_probe] = self.indices[probe];
                        self.indices[probe] = Pos::none();
                    } else {
                        break;
                    }
                } else {
                    break;
                }
                last_probe = probe;
            });
        }

        (entry.key, entry.value)
    }

}

use std::slice::Iter as SliceIter;
use std::slice::IterMut as SliceIterMut;
use std::vec::IntoIter as VecIntoIter;

pub struct Keys<'a, K: 'a, V: 'a> {
    iter: SliceIter<'a, Bucket<K, V>>,
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<&'a K> {
        self.iter.next().map(|ent| &ent.key)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    fn count(self) -> usize {
        self.iter.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth(n).map(|ent| &ent.key)
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, K, V> DoubleEndedIterator for Keys<'a, K, V> {
    fn next_back(&mut self) -> Option<&'a K> {
        self.iter.next_back().map(|ent| &ent.key)
    }
}

impl<'a, K, V> ExactSizeIterator for Keys<'a, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

pub struct Values<'a, K: 'a, V: 'a> {
    iter: SliceIter<'a, Bucket<K, V>>,
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|ent| &ent.value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    fn count(self) -> usize {
        self.iter.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth(n).map(|ent| &ent.value)
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, K, V> DoubleEndedIterator for Values<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|ent| &ent.value)
    }
}

impl<'a, K, V> ExactSizeIterator for Values<'a, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

pub struct ValuesMut<'a, K: 'a, V: 'a> {
    iter: SliceIterMut<'a, Bucket<K, V>>,
}

impl<'a, K, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|ent| &mut ent.value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    fn count(self) -> usize {
        self.iter.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth(n).map(|ent| &mut ent.value)
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, K, V> DoubleEndedIterator for ValuesMut<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|ent| &mut ent.value)
    }
}

impl<'a, K, V> ExactSizeIterator for ValuesMut<'a, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

pub struct Iter<'a, K: 'a, V: 'a> {
    iter: SliceIter<'a, Bucket<K, V>>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|e| (&e.key, &e.value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    fn count(self) -> usize {
        self.iter.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth(n).map(|e| (&e.key, &e.value))
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, K, V> DoubleEndedIterator for Iter<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|e| (&e.key, &e.value))
    }
}

impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

pub struct IterMut<'a, K: 'a, V: 'a> {
    iter: SliceIterMut<'a, Bucket<K, V>>,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|e| (&e.key, &mut e.value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    fn count(self) -> usize {
        self.iter.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth(n).map(|e| (&e.key, &mut e.value))
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, K, V> DoubleEndedIterator for IterMut<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|e| (&e.key, &mut e.value))
    }
}

impl<'a, K, V> ExactSizeIterator for IterMut<'a, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

pub struct IntoIter<K, V> {
    iter: VecIntoIter<Bucket<K, V>>,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|e| (e.key, e.value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    fn count(self) -> usize {
        self.iter.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth(n).map(|e| (e.key, e.value))
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, K, V> DoubleEndedIterator for IntoIter<K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|e| (e.key, e.value))
    }
}

impl<K, V> ExactSizeIterator for IntoIter<K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}


impl<'a, K, V, S> IntoIterator for &'a OrderMap<K, V, S>
    where K: Hash + Eq,
          S: BuildHasher,
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V, S> IntoIterator for &'a mut OrderMap<K, V, S>
    where K: Hash + Eq,
          S: BuildHasher,
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K, V, S> IntoIterator for OrderMap<K, V, S>
    where K: Hash + Eq,
          S: BuildHasher,
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            iter: self.entries.into_iter(),
        }
    }
}

use std::ops::{Index, IndexMut};

impl<'a, K, V, Q: ?Sized, S> Index<&'a Q> for OrderMap<K, V, S>
    where Q: Hash + Equivalent<K>,
          K: Hash + Eq,
          S: BuildHasher,
{
    type Output = V;

    /// ***Panics*** if `key` is not present in the map.
    fn index(&self, key: &'a Q) -> &V {
        if let Some(v) = self.get(key) {
            v
        } else {
            panic!("OrderMap: key not found")
        }
    }
}

/// Mutable indexing allows changing / updating values of key-value
/// pairs that are already present.
///
/// You can **not** insert new pairs with index syntax, use `.insert()`.
impl<'a, K, V, Q: ?Sized, S> IndexMut<&'a Q> for OrderMap<K, V, S>
    where Q: Hash + Equivalent<K>,
          K: Hash + Eq,
          S: BuildHasher,
{
    /// ***Panics*** if `key` is not present in the map.
    fn index_mut(&mut self, key: &'a Q) -> &mut V {
        if let Some(v) = self.get_mut(key) {
            v
        } else {
            panic!("OrderMap: key not found")
        }
    }
}

use std::iter::FromIterator;

impl<K, V, S> FromIterator<(K, V)> for OrderMap<K, V, S>
    where K: Hash + Eq,
          S: BuildHasher + Default,
{
    fn from_iter<I: IntoIterator<Item=(K, V)>>(iterable: I) -> Self {
        let iter = iterable.into_iter();
        let (low, _) = iter.size_hint();
        let mut map = Self::with_capacity_and_hasher(low, <_>::default());
        for (k, v) in iter { map.insert(k, v); }
        map
    }
}

impl<K, V, S> Extend<(K, V)> for OrderMap<K, V, S>
    where K: Hash + Eq,
          S: BuildHasher,
{
    fn extend<I: IntoIterator<Item=(K, V)>>(&mut self, iterable: I) {
        for (k, v) in iterable { self.insert(k, v); }
    }
}

impl<'a, K, V, S> Extend<(&'a K, &'a V)> for OrderMap<K, V, S>
    where K: Hash + Eq + Copy,
          V: Copy,
          S: BuildHasher,
{
    fn extend<I: IntoIterator<Item=(&'a K, &'a V)>>(&mut self, iterable: I) {
        self.extend(iterable.into_iter().map(|(&key, &value)| (key, value)));
    }
}

impl<K, V, S> Default for OrderMap<K, V, S>
    where S: BuildHasher + Default,
{
    fn default() -> Self {
        Self::with_capacity_and_hasher(0, S::default())
    }
}

impl<K, V1, S1, V2, S2> PartialEq<OrderMap<K, V2, S2>> for OrderMap<K, V1, S1>
    where K: Hash + Eq,
          V1: PartialEq<V2>,
          S1: BuildHasher,
          S2: BuildHasher
{
    fn eq(&self, other: &OrderMap<K, V2, S2>) -> bool {
        if self.len() != other.len() {
            return false;
        }

        self.iter().all(|(key, value)| other.get(key).map_or(false, |v| *value == *v))
    }
}

impl<K, V, S> Eq for OrderMap<K, V, S>
    where K: Eq + Hash,
          V: Eq,
          S: BuildHasher
{
}

pub struct Drain<'a, K, V> where K: 'a, V: 'a {
    inner: ::std::vec::Drain<'a, Bucket<K, V>>
}

impl<'a, K, V> Iterator for Drain<'a, K, V> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|bucket| (bucket.key, bucket.value))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use util::enumerate;
//
//     #[test]
//     fn it_works() {
//         let mut map = OrderMap::new();
//         assert_eq!(map.is_empty(), true);
//         map.insert(1, ());
//         map.insert(1, ());
//         assert_eq!(map.len(), 1);
//         assert!(map.get(&1).is_some());
//         assert_eq!(map.is_empty(), false);
//     }
//
//     #[test]
//     fn new() {
//         let map = OrderMap::<String, String>::new();
//         println!("{:?}", map);
//         assert_eq!(map.capacity(), 0);
//         assert_eq!(map.len(), 0);
//         assert_eq!(map.is_empty(), true);
//     }
//
//     #[test]
//     fn insert() {
//         let insert = [0, 4, 2, 12, 8, 7, 11, 5];
//         let not_present = [1, 3, 6, 9, 10];
//         let mut map = OrderMap::with_capacity(insert.len());
//
//         for (i, &elt) in enumerate(&insert) {
//             assert_eq!(map.len(), i);
//             map.insert(elt, elt);
//             assert_eq!(map.len(), i + 1);
//             assert_eq!(map.get(&elt), Some(&elt));
//             assert_eq!(map[&elt], elt);
//         }
//         println!("{:?}", map);
//
//         for &elt in &not_present {
//             assert!(map.get(&elt).is_none());
//         }
//     }
//
//     #[test]
//     fn insert_2() {
//         let mut map = OrderMap::with_capacity(16);
//
//         let mut keys = vec![];
//         keys.extend(0..16);
//         keys.extend(128..267);
//
//         for &i in &keys {
//             let old_map = map.clone();
//             map.insert(i, ());
//             for key in old_map.keys() {
//                 if !map.get(key).is_some() {
//                     println!("old_map: {:?}", old_map);
//                     println!("map: {:?}", map);
//                     panic!("did not find {} in map", key);
//                 }
//             }
//         }
//
//         for &i in &keys {
//             assert!(map.get(&i).is_some(), "did not find {}", i);
//         }
//     }
//
//     #[test]
//     fn insert_order() {
//         let insert = [0, 4, 2, 12, 8, 7, 11, 5, 3, 17, 19, 22, 23];
//         let mut map = OrderMap::new();
//
//         for &elt in &insert {
//             map.insert(elt, ());
//         }
//
//         assert_eq!(map.keys().count(), map.len());
//         assert_eq!(map.keys().count(), insert.len());
//         for (a, b) in insert.iter().zip(map.keys()) {
//             assert_eq!(a, b);
//         }
//         for (i, k) in (0..insert.len()).zip(map.keys()) {
//             assert_eq!(map.get_index(i).unwrap().0, k);
//         }
//     }
//
//     #[test]
//     fn grow() {
//         let insert = [0, 4, 2, 12, 8, 7, 11];
//         let not_present = [1, 3, 6, 9, 10];
//         let mut map = OrderMap::with_capacity(insert.len());
//
//
//         for (i, &elt) in enumerate(&insert) {
//             assert_eq!(map.len(), i);
//             map.insert(elt, elt);
//             assert_eq!(map.len(), i + 1);
//             assert_eq!(map.get(&elt), Some(&elt));
//             assert_eq!(map[&elt], elt);
//         }
//
//         println!("{:?}", map);
//         for &elt in &insert {
//             map.insert(elt * 10, elt);
//         }
//         for &elt in &insert {
//             map.insert(elt * 100, elt);
//         }
//         for (i, &elt) in insert.iter().cycle().enumerate().take(100) {
//             map.insert(elt * 100 + i as i32, elt);
//         }
//         println!("{:?}", map);
//         for &elt in &not_present {
//             assert!(map.get(&elt).is_none());
//         }
//     }
//
//     #[test]
//     fn remove() {
//         let insert = [0, 4, 2, 12, 8, 7, 11, 5, 3, 17, 19, 22, 23];
//         let mut map = OrderMap::new();
//
//         for &elt in &insert {
//             map.insert(elt, elt);
//         }
//
//         assert_eq!(map.keys().count(), map.len());
//         assert_eq!(map.keys().count(), insert.len());
//         for (a, b) in insert.iter().zip(map.keys()) {
//             assert_eq!(a, b);
//         }
//
//         let remove_fail = [99, 77];
//         let remove = [4, 12, 8, 7];
//
//         for &key in &remove_fail {
//             assert!(map.swap_remove_full(&key).is_none());
//         }
//         println!("{:?}", map);
//         for &key in &remove {
//         //println!("{:?}", map);
//             let index = map.get_full(&key).unwrap().0;
//             assert_eq!(map.swap_remove_full(&key), Some((index, key, key)));
//         }
//         println!("{:?}", map);
//
//         for key in &insert {
//             assert_eq!(map.get(key).is_some(), !remove.contains(key));
//         }
//         assert_eq!(map.len(), insert.len() - remove.len());
//         assert_eq!(map.keys().count(), insert.len() - remove.len());
//     }
//
//     #[test]
//     fn swap_remove_index() {
//         let insert = [0, 4, 2, 12, 8, 7, 11, 5, 3, 17, 19, 22, 23];
//         let mut map = OrderMap::new();
//
//         for &elt in &insert {
//             map.insert(elt, elt * 2);
//         }
//
//         let mut vector = insert.to_vec();
//         let remove_sequence = &[3, 3, 10, 4, 5, 4, 3, 0, 1];
//
//         // check that the same swap remove sequence on vec and map
//         // have the same result.
//         for &rm in remove_sequence {
//             let out_vec = vector.swap_remove(rm);
//             let (out_map, _) = map.swap_remove_index(rm).unwrap();
//             assert_eq!(out_vec, out_map);
//         }
//         assert_eq!(vector.len(), map.len());
//         for (a, b) in vector.iter().zip(map.keys()) {
//             assert_eq!(a, b);
//         }
//     }
//
//     #[test]
//     fn partial_eq_and_eq() {
//         let mut map_a = OrderMap::new();
//         map_a.insert(1, "1");
//         map_a.insert(2, "2");
//         let mut map_b = map_a.clone();
//         assert_eq!(map_a, map_b);
//         map_b.remove(&1);
//         assert_ne!(map_a, map_b);
//
//         let map_c: OrderMap<_, String> = map_b.into_iter().map(|(k, v)| (k, v.to_owned())).collect();
//         assert_ne!(map_a, map_c);
//         assert_ne!(map_c, map_a);
//     }
//
//     #[test]
//     fn extend() {
//         let mut map = OrderMap::new();
//         map.extend(vec![(&1, &2), (&3, &4)]);
//         map.extend(vec![(5, 6)]);
//         assert_eq!(map.into_iter().collect::<Vec<_>>(), vec![(1, 2), (3, 4), (5, 6)]);
//     }
//
//     #[test]
//     fn entry() {
//         let mut map = OrderMap::new();
//
//         map.insert(1, "1");
//         map.insert(2, "2");
//         {
//             let e = map.entry(3);
//             assert_eq!(e.index(), 2);
//             let e = e.or_insert("3");
//             assert_eq!(e, &"3");
//         }
//
//         let e = map.entry(2);
//         assert_eq!(e.index(), 1);
//         assert_eq!(e.key(), &2);
//         match e {
//             Entry::Occupied(ref e) => assert_eq!(e.get(), &"2"),
//             Entry::Vacant(_) => panic!()
//         }
//         assert_eq!(e.or_insert("4"), &"2");
//     }
// }
