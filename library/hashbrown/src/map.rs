use crate::raw::{Allocator, Bucket, Global, RawDrain, RawIntoIter, RawIter, RawTable};
use crate::TryReserveError;
#[cfg(feature = "nightly")]
use crate::UnavailableMutError;
use core::borrow::Borrow;
use core::fmt::{self, Debug};
use core::hash::{BuildHasher, Hash};
use core::iter::{FromIterator, FusedIterator};
use core::marker::PhantomData;
use core::mem;
#[cfg(feature = "nightly")]
use core::mem::MaybeUninit;
use core::ops::Index;

/// Default hasher for `HashMap`.
#[cfg(feature = "ahash")]
pub type DefaultHashBuilder = ahash::RandomState;

/// Dummy default hasher for `HashMap`.
#[cfg(not(feature = "ahash"))]
pub enum DefaultHashBuilder {}

/// A hash map implemented with quadratic probing and SIMD lookup.
///
/// The default hashing algorithm is currently [`AHash`], though this is
/// subject to change at any point in the future. This hash function is very
/// fast for all types of keys, but this algorithm will typically *not* protect
/// against attacks such as HashDoS.
///
/// The hashing algorithm can be replaced on a per-`HashMap` basis using the
/// [`default`], [`with_hasher`], and [`with_capacity_and_hasher`] methods. Many
/// alternative algorithms are available on crates.io, such as the [`fnv`] crate.
///
/// It is required that the keys implement the [`Eq`] and [`Hash`] traits, although
/// this can frequently be achieved by using `#[derive(PartialEq, Eq, Hash)]`.
/// If you implement these yourself, it is important that the following
/// property holds:
///
/// ```text
/// k1 == k2 -> hash(k1) == hash(k2)
/// ```
///
/// In other words, if two keys are equal, their hashes must be equal.
///
/// It is a logic error for a key to be modified in such a way that the key's
/// hash, as determined by the [`Hash`] trait, or its equality, as determined by
/// the [`Eq`] trait, changes while it is in the map. This is normally only
/// possible through [`Cell`], [`RefCell`], global state, I/O, or unsafe code.
///
/// It is also a logic error for the [`Hash`] implementation of a key to panic.
/// This is generally only possible if the trait is implemented manually. If a
/// panic does occur then the contents of the `HashMap` may become corrupted and
/// some items may be dropped from the table.
///
/// # Examples
///
/// ```
/// use hashbrown::HashMap;
///
/// // Type inference lets us omit an explicit type signature (which
/// // would be `HashMap<String, String>` in this example).
/// let mut book_reviews = HashMap::new();
///
/// // Review some books.
/// book_reviews.insert(
///     "Adventures of Huckleberry Finn".to_string(),
///     "My favorite book.".to_string(),
/// );
/// book_reviews.insert(
///     "Grimms' Fairy Tales".to_string(),
///     "Masterpiece.".to_string(),
/// );
/// book_reviews.insert(
///     "Pride and Prejudice".to_string(),
///     "Very enjoyable.".to_string(),
/// );
/// book_reviews.insert(
///     "The Adventures of Sherlock Holmes".to_string(),
///     "Eye lyked it alot.".to_string(),
/// );
///
/// // Check for a specific one.
/// // When collections store owned values (String), they can still be
/// // queried using references (&str).
/// if !book_reviews.contains_key("Les Misérables") {
///     println!("We've got {} reviews, but Les Misérables ain't one.",
///              book_reviews.len());
/// }
///
/// // oops, this review has a lot of spelling mistakes, let's delete it.
/// book_reviews.remove("The Adventures of Sherlock Holmes");
///
/// // Look up the values associated with some keys.
/// let to_find = ["Pride and Prejudice", "Alice's Adventure in Wonderland"];
/// for &book in &to_find {
///     match book_reviews.get(book) {
///         Some(review) => println!("{}: {}", book, review),
///         None => println!("{} is unreviewed.", book)
///     }
/// }
///
/// // Look up the value for a key (will panic if the key is not found).
/// println!("Review for Jane: {}", book_reviews["Pride and Prejudice"]);
///
/// // Iterate over everything.
/// for (book, review) in &book_reviews {
///     println!("{}: \"{}\"", book, review);
/// }
/// ```
///
/// `HashMap` also implements an [`Entry API`](#method.entry), which allows
/// for more complex methods of getting, setting, updating and removing keys and
/// their values:
///
/// ```
/// use hashbrown::HashMap;
///
/// // type inference lets us omit an explicit type signature (which
/// // would be `HashMap<&str, u8>` in this example).
/// let mut player_stats = HashMap::new();
///
/// fn random_stat_buff() -> u8 {
///     // could actually return some random value here - let's just return
///     // some fixed value for now
///     42
/// }
///
/// // insert a key only if it doesn't already exist
/// player_stats.entry("health").or_insert(100);
///
/// // insert a key using a function that provides a new value only if it
/// // doesn't already exist
/// player_stats.entry("defence").or_insert_with(random_stat_buff);
///
/// // update a key, guarding against the key possibly not being set
/// let stat = player_stats.entry("attack").or_insert(100);
/// *stat += random_stat_buff();
/// ```
///
/// The easiest way to use `HashMap` with a custom key type is to derive [`Eq`] and [`Hash`].
/// We must also derive [`PartialEq`].
///
/// [`Eq`]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
/// [`Hash`]: https://doc.rust-lang.org/std/hash/trait.Hash.html
/// [`PartialEq`]: https://doc.rust-lang.org/std/cmp/trait.PartialEq.html
/// [`RefCell`]: https://doc.rust-lang.org/std/cell/struct.RefCell.html
/// [`Cell`]: https://doc.rust-lang.org/std/cell/struct.Cell.html
/// [`default`]: #method.default
/// [`with_hasher`]: #method.with_hasher
/// [`with_capacity_and_hasher`]: #method.with_capacity_and_hasher
/// [`fnv`]: https://crates.io/crates/fnv
/// [`AHash`]: https://crates.io/crates/ahash
///
/// ```
/// use hashbrown::HashMap;
///
/// #[derive(Hash, Eq, PartialEq, Debug)]
/// struct Viking {
///     name: String,
///     country: String,
/// }
///
/// impl Viking {
///     /// Creates a new Viking.
///     fn new(name: &str, country: &str) -> Viking {
///         Viking { name: name.to_string(), country: country.to_string() }
///     }
/// }
///
/// // Use a HashMap to store the vikings' health points.
/// let mut vikings = HashMap::new();
///
/// vikings.insert(Viking::new("Einar", "Norway"), 25);
/// vikings.insert(Viking::new("Olaf", "Denmark"), 24);
/// vikings.insert(Viking::new("Harald", "Iceland"), 12);
///
/// // Use derived implementation to print the status of the vikings.
/// for (viking, health) in &vikings {
///     println!("{:?} has {} hp", viking, health);
/// }
/// ```
///
/// A `HashMap` with fixed list of elements can be initialized from an array:
///
/// ```
/// use hashbrown::HashMap;
///
/// let timber_resources: HashMap<&str, i32> = [("Norway", 100), ("Denmark", 50), ("Iceland", 10)]
///     .iter().cloned().collect();
/// // use the values stored in map
/// ```
pub struct HashMap<K, V, S = DefaultHashBuilder, A: Allocator + Clone = Global> {
    pub(crate) hash_builder: S,
    pub(crate) table: RawTable<(K, V), A>,
}

impl<K: Clone, V: Clone, S: Clone, A: Allocator + Clone> Clone for HashMap<K, V, S, A> {
    fn clone(&self) -> Self {
        HashMap {
            hash_builder: self.hash_builder.clone(),
            table: self.table.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.table.clone_from(&source.table);

        // Update hash_builder only if we successfully cloned all elements.
        self.hash_builder.clone_from(&source.hash_builder);
    }
}

/// Ensures that a single closure type across uses of this which, in turn prevents multiple
/// instances of any functions like RawTable::reserve from being generated
#[cfg_attr(feature = "inline-more", inline)]
pub(crate) fn make_hasher<K, Q, V, S>(hash_builder: &S) -> impl Fn(&(Q, V)) -> u64 + '_
where
    K: Borrow<Q>,
    Q: Hash,
    S: BuildHasher,
{
    move |val| make_hash::<K, Q, S>(hash_builder, &val.0)
}

/// Ensures that a single closure type across uses of this which, in turn prevents multiple
/// instances of any functions like RawTable::reserve from being generated
#[cfg_attr(feature = "inline-more", inline)]
fn equivalent_key<Q, K, V>(k: &Q) -> impl Fn(&(K, V)) -> bool + '_
where
    K: Borrow<Q>,
    Q: ?Sized + Eq,
{
    move |x| k.eq(x.0.borrow())
}

/// Ensures that a single closure type across uses of this which, in turn prevents multiple
/// instances of any functions like RawTable::reserve from being generated
#[cfg_attr(feature = "inline-more", inline)]
fn equivalent<Q, K>(k: &Q) -> impl Fn(&K) -> bool + '_
where
    K: Borrow<Q>,
    Q: ?Sized + Eq,
{
    move |x| k.eq(x.borrow())
}

#[cfg_attr(feature = "inline-more", inline)]
pub(crate) fn make_hash<K, Q, S>(hash_builder: &S, val: &Q) -> u64
where
    K: Borrow<Q>,
    Q: Hash + ?Sized,
    S: BuildHasher,
{
    use core::hash::Hasher;
    let mut state = hash_builder.build_hasher();
    val.hash(&mut state);
    state.finish()
}

#[cfg_attr(feature = "inline-more", inline)]
pub(crate) fn make_insert_hash<K, S>(hash_builder: &S, val: &K) -> u64
where
    K: Hash,
    S: BuildHasher,
{
    use core::hash::Hasher;
    let mut state = hash_builder.build_hasher();
    val.hash(&mut state);
    state.finish()
}

#[cfg(feature = "ahash")]
impl<K, V> HashMap<K, V, DefaultHashBuilder> {
    /// Creates an empty `HashMap`.
    ///
    /// The hash map is initially created with a capacity of 0, so it will not allocate until it
    /// is first inserted into.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// let mut map: HashMap<&str, i32> = HashMap::new();
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty `HashMap` with the specified capacity.
    ///
    /// The hash map will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is 0, the hash map will not allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// let mut map: HashMap<&str, i32> = HashMap::with_capacity(10);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, DefaultHashBuilder::default())
    }
}

#[cfg(feature = "ahash")]
impl<K, V, A: Allocator + Clone> HashMap<K, V, DefaultHashBuilder, A> {
    /// Creates an empty `HashMap` using the given allocator.
    ///
    /// The hash map is initially created with a capacity of 0, so it will not allocate until it
    /// is first inserted into.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn new_in(alloc: A) -> Self {
        Self::with_hasher_in(DefaultHashBuilder::default(), alloc)
    }

    /// Creates an empty `HashMap` with the specified capacity using the given allocator.
    ///
    /// The hash map will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is 0, the hash map will not allocate.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        Self::with_capacity_and_hasher_in(capacity, DefaultHashBuilder::default(), alloc)
    }
}

impl<K, V, S> HashMap<K, V, S> {
    /// Creates an empty `HashMap` which will use the given hash builder to hash
    /// keys.
    ///
    /// The created map has the default initial capacity.
    ///
    /// Warning: `hash_builder` is normally randomly generated, and
    /// is designed to allow HashMaps to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting it
    /// manually using this function can expose a DoS attack vector.
    ///
    /// The `hash_builder` passed should implement the [`BuildHasher`] trait for
    /// the HashMap to be useful, see its documentation for details.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// use hashbrown::hash_map::DefaultHashBuilder;
    ///
    /// let s = DefaultHashBuilder::default();
    /// let mut map = HashMap::with_hasher(s);
    /// map.insert(1, 2);
    /// ```
    ///
    /// [`BuildHasher`]: ../../std/hash/trait.BuildHasher.html
    #[cfg_attr(feature = "inline-more", inline)]
    pub const fn with_hasher(hash_builder: S) -> Self {
        Self {
            hash_builder,
            table: RawTable::new(),
        }
    }

    /// Creates an empty `HashMap` with the specified capacity, using `hash_builder`
    /// to hash the keys.
    ///
    /// The hash map will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is 0, the hash map will not allocate.
    ///
    /// Warning: `hash_builder` is normally randomly generated, and
    /// is designed to allow HashMaps to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting it
    /// manually using this function can expose a DoS attack vector.
    ///
    /// The `hash_builder` passed should implement the [`BuildHasher`] trait for
    /// the HashMap to be useful, see its documentation for details.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// use hashbrown::hash_map::DefaultHashBuilder;
    ///
    /// let s = DefaultHashBuilder::default();
    /// let mut map = HashMap::with_capacity_and_hasher(10, s);
    /// map.insert(1, 2);
    /// ```
    ///
    /// [`BuildHasher`]: ../../std/hash/trait.BuildHasher.html
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        Self {
            hash_builder,
            table: RawTable::with_capacity(capacity),
        }
    }
}

impl<K, V, S, A: Allocator + Clone> HashMap<K, V, S, A> {
    /// Creates an empty `HashMap` which will use the given hash builder to hash
    /// keys. It will be allocated with the given allocator.
    ///
    /// The created map has the default initial capacity.
    ///
    /// Warning: `hash_builder` is normally randomly generated, and
    /// is designed to allow HashMaps to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting it
    /// manually using this function can expose a DoS attack vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// use hashbrown::hash_map::DefaultHashBuilder;
    ///
    /// let s = DefaultHashBuilder::default();
    /// let mut map = HashMap::with_hasher(s);
    /// map.insert(1, 2);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn with_hasher_in(hash_builder: S, alloc: A) -> Self {
        Self {
            hash_builder,
            table: RawTable::new_in(alloc),
        }
    }

    /// Creates an empty `HashMap` with the specified capacity, using `hash_builder`
    /// to hash the keys. It will be allocated with the given allocator.
    ///
    /// The hash map will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is 0, the hash map will not allocate.
    ///
    /// Warning: `hash_builder` is normally randomly generated, and
    /// is designed to allow HashMaps to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting it
    /// manually using this function can expose a DoS attack vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// use hashbrown::hash_map::DefaultHashBuilder;
    ///
    /// let s = DefaultHashBuilder::default();
    /// let mut map = HashMap::with_capacity_and_hasher(10, s);
    /// map.insert(1, 2);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn with_capacity_and_hasher_in(capacity: usize, hash_builder: S, alloc: A) -> Self {
        Self {
            hash_builder,
            table: RawTable::with_capacity_in(capacity, alloc),
        }
    }

    /// Returns a reference to the map's [`BuildHasher`].
    ///
    /// [`BuildHasher`]: https://doc.rust-lang.org/std/hash/trait.BuildHasher.html
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// use hashbrown::hash_map::DefaultHashBuilder;
    ///
    /// let hasher = DefaultHashBuilder::default();
    /// let map: HashMap<i32, i32> = HashMap::with_hasher(hasher);
    /// let hasher: &DefaultHashBuilder = map.hasher();
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }

    /// Returns the number of elements the map can hold without reallocating.
    ///
    /// This number is a lower bound; the `HashMap<K, V>` might be able to hold
    /// more, but is guaranteed to be able to hold at least this many.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// let map: HashMap<i32, i32> = HashMap::with_capacity(100);
    /// assert!(map.capacity() >= 100);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn capacity(&self) -> usize {
        self.table.capacity()
    }

    /// An iterator visiting all keys in arbitrary order.
    /// The iterator element type is `&'a K`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for key in map.keys() {
    ///     println!("{}", key);
    /// }
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys { inner: self.iter() }
    }

    /// An iterator visiting all values in arbitrary order.
    /// The iterator element type is `&'a V`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for val in map.values() {
    ///     println!("{}", val);
    /// }
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn values(&self) -> Values<'_, K, V> {
        Values { inner: self.iter() }
    }

    /// An iterator visiting all values mutably in arbitrary order.
    /// The iterator element type is `&'a mut V`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map = HashMap::new();
    ///
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for val in map.values_mut() {
    ///     *val = *val + 10;
    /// }
    ///
    /// for val in map.values() {
    ///     println!("{}", val);
    /// }
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut {
            inner: self.iter_mut(),
        }
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    /// The iterator element type is `(&'a K, &'a V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for (key, val) in map.iter() {
    ///     println!("key: {} val: {}", key, val);
    /// }
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn iter(&self) -> Iter<'_, K, V> {
        // Here we tie the lifetime of self to the iter.
        unsafe {
            Iter {
                inner: self.table.iter(),
                marker: PhantomData,
            }
        }
    }

    /// An iterator visiting all key-value pairs in arbitrary order,
    /// with mutable references to the values.
    /// The iterator element type is `(&'a K, &'a mut V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// // Update all values
    /// for (_, val) in map.iter_mut() {
    ///     *val *= 2;
    /// }
    ///
    /// for (key, val) in &map {
    ///     println!("key: {} val: {}", key, val);
    /// }
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        // Here we tie the lifetime of self to the iter.
        unsafe {
            IterMut {
                inner: self.table.iter(),
                marker: PhantomData,
            }
        }
    }

    #[cfg(test)]
    #[cfg_attr(feature = "inline-more", inline)]
    fn raw_capacity(&self) -> usize {
        self.table.buckets()
    }

    /// Returns the number of elements in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut a = HashMap::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1, "a");
    /// assert_eq!(a.len(), 1);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Returns `true` if the map contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut a = HashMap::new();
    /// assert!(a.is_empty());
    /// a.insert(1, "a");
    /// assert!(!a.is_empty());
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears the map, returning all key-value pairs as an iterator. Keeps the
    /// allocated memory for reuse.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut a = HashMap::new();
    /// a.insert(1, "a");
    /// a.insert(2, "b");
    ///
    /// for (k, v) in a.drain().take(1) {
    ///     assert!(k == 1 || k == 2);
    ///     assert!(v == "a" || v == "b");
    /// }
    ///
    /// assert!(a.is_empty());
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn drain(&mut self) -> Drain<'_, K, V, A> {
        Drain {
            inner: self.table.drain(),
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all pairs `(k, v)` such that `f(&k,&mut v)` returns `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map: HashMap<i32, i32> = (0..8).map(|x|(x, x*10)).collect();
    /// map.retain(|&k, _| k % 2 == 0);
    /// assert_eq!(map.len(), 4);
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        // Here we only use `iter` as a temporary, preventing use-after-free
        unsafe {
            for item in self.table.iter() {
                let &mut (ref key, ref mut value) = item.as_mut();
                if !f(key, value) {
                    self.table.erase(item);
                }
            }
        }
    }

    /// Drains elements which are true under the given predicate,
    /// and returns an iterator over the removed items.
    ///
    /// In other words, move all pairs `(k, v)` such that `f(&k,&mut v)` returns `true` out
    /// into another iterator.
    ///
    /// When the returned DrainedFilter is dropped, any remaining elements that satisfy
    /// the predicate are dropped from the table.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map: HashMap<i32, i32> = (0..8).map(|x| (x, x)).collect();
    /// let drained: HashMap<i32, i32> = map.drain_filter(|k, _v| k % 2 == 0).collect();
    ///
    /// let mut evens = drained.keys().cloned().collect::<Vec<_>>();
    /// let mut odds = map.keys().cloned().collect::<Vec<_>>();
    /// evens.sort();
    /// odds.sort();
    ///
    /// assert_eq!(evens, vec![0, 2, 4, 6]);
    /// assert_eq!(odds, vec![1, 3, 5, 7]);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn drain_filter<F>(&mut self, f: F) -> DrainFilter<'_, K, V, F, A>
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        DrainFilter {
            f,
            inner: DrainFilterInner {
                iter: unsafe { self.table.iter() },
                table: &mut self.table,
            },
        }
    }

    /// Clears the map, removing all key-value pairs. Keeps the allocated memory
    /// for reuse.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut a = HashMap::new();
    /// a.insert(1, "a");
    /// a.clear();
    /// assert!(a.is_empty());
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn clear(&mut self) {
        self.table.clear();
    }
}

impl<K, V, S, A> HashMap<K, V, S, A>
where
    K: Eq + Hash,
    S: BuildHasher,
    A: Allocator + Clone,
{
    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the `HashMap`. The collection may reserve more space to avoid
    /// frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new allocation size overflows [`usize`].
    ///
    /// [`usize`]: https://doc.rust-lang.org/std/primitive.usize.html
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// let mut map: HashMap<&str, i32> = HashMap::new();
    /// map.reserve(10);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn reserve(&mut self, additional: usize) {
        self.table
            .reserve(additional, make_hasher::<K, _, V, S>(&self.hash_builder));
    }

    /// Tries to reserve capacity for at least `additional` more elements to be inserted
    /// in the given `HashMap<K,V>`. The collection may reserve more space to avoid
    /// frequent reallocations.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// let mut map: HashMap<&str, isize> = HashMap::new();
    /// map.try_reserve(10).expect("why is the test harness OOMing on 10 bytes?");
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.table
            .try_reserve(additional, make_hasher::<K, _, V, S>(&self.hash_builder))
    }

    /// Shrinks the capacity of the map as much as possible. It will drop
    /// down as much as possible while maintaining the internal rules
    /// and possibly leaving some space in accordance with the resize policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map: HashMap<i32, i32> = HashMap::with_capacity(100);
    /// map.insert(1, 2);
    /// map.insert(3, 4);
    /// assert!(map.capacity() >= 100);
    /// map.shrink_to_fit();
    /// assert!(map.capacity() >= 2);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn shrink_to_fit(&mut self) {
        self.table
            .shrink_to(0, make_hasher::<K, _, V, S>(&self.hash_builder));
    }

    /// Shrinks the capacity of the map with a lower limit. It will drop
    /// down no lower than the supplied limit while maintaining the internal rules
    /// and possibly leaving some space in accordance with the resize policy.
    ///
    /// This function does nothing if the current capacity is smaller than the
    /// supplied minimum capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map: HashMap<i32, i32> = HashMap::with_capacity(100);
    /// map.insert(1, 2);
    /// map.insert(3, 4);
    /// assert!(map.capacity() >= 100);
    /// map.shrink_to(10);
    /// assert!(map.capacity() >= 10);
    /// map.shrink_to(0);
    /// assert!(map.capacity() >= 2);
    /// map.shrink_to(10);
    /// assert!(map.capacity() >= 2);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.table
            .shrink_to(min_capacity, make_hasher::<K, _, V, S>(&self.hash_builder));
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut letters = HashMap::new();
    ///
    /// for ch in "a short treatise on fungi".chars() {
    ///     let counter = letters.entry(ch).or_insert(0);
    ///     *counter += 1;
    /// }
    ///
    /// assert_eq!(letters[&'s'], 2);
    /// assert_eq!(letters[&'t'], 3);
    /// assert_eq!(letters[&'u'], 1);
    /// assert_eq!(letters.get(&'y'), None);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, S, A> {
        let hash = make_insert_hash::<K, S>(&self.hash_builder, &key);
        if let Some(elem) = self.table.find(hash, equivalent_key(&key)) {
            Entry::Occupied(OccupiedEntry {
                hash,
                key: Some(key),
                elem,
                table: self,
            })
        } else {
            Entry::Vacant(VacantEntry {
                hash,
                key,
                table: self,
            })
        }
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
    /// [`Hash`]: https://doc.rust-lang.org/std/hash/trait.Hash.html
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.get(&2), None);
    /// ```
    #[inline]
    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.get_inner(k) {
            Some(&(_, ref v)) => Some(v),
            None => None,
        }
    }

    /// Returns the key-value pair corresponding to the supplied key.
    ///
    /// The supplied key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
    /// [`Hash`]: https://doc.rust-lang.org/std/hash/trait.Hash.html
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get_key_value(&1), Some((&1, &"a")));
    /// assert_eq!(map.get_key_value(&2), None);
    /// ```
    #[inline]
    pub fn get_key_value<Q: ?Sized>(&self, k: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.get_inner(k) {
            Some(&(ref key, ref value)) => Some((key, value)),
            None => None,
        }
    }

    #[inline]
    fn get_inner<Q: ?Sized>(&self, k: &Q) -> Option<&(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let hash = make_hash::<K, Q, S>(&self.hash_builder, k);
        self.table.get(hash, equivalent_key(k))
    }

    /// Returns the key-value pair corresponding to the supplied key, with a mutable reference to value.
    ///
    /// The supplied key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
    /// [`Hash`]: https://doc.rust-lang.org/std/hash/trait.Hash.html
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// let (k, v) = map.get_key_value_mut(&1).unwrap();
    /// assert_eq!(k, &1);
    /// assert_eq!(v, &mut "a");
    /// *v = "b";
    /// assert_eq!(map.get_key_value_mut(&1), Some((&1, &mut "b")));
    /// assert_eq!(map.get_key_value_mut(&2), None);
    /// ```
    #[inline]
    pub fn get_key_value_mut<Q: ?Sized>(&mut self, k: &Q) -> Option<(&K, &mut V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.get_inner_mut(k) {
            Some(&mut (ref key, ref mut value)) => Some((key, value)),
            None => None,
        }
    }

    /// Returns `true` if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
    /// [`Hash`]: https://doc.rust-lang.org/std/hash/trait.Hash.html
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn contains_key<Q: ?Sized>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get_inner(k).is_some()
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
    /// [`Hash`]: https://doc.rust-lang.org/std/hash/trait.Hash.html
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// if let Some(x) = map.get_mut(&1) {
    ///     *x = "b";
    /// }
    /// assert_eq!(map[&1], "b");
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn get_mut<Q: ?Sized>(&mut self, k: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.get_inner_mut(k) {
            Some(&mut (_, ref mut v)) => Some(v),
            None => None,
        }
    }

    #[inline]
    fn get_inner_mut<Q: ?Sized>(&mut self, k: &Q) -> Option<&mut (K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let hash = make_hash::<K, Q, S>(&self.hash_builder, k);
        self.table.get_mut(hash, equivalent_key(k))
    }

    /// Attempts to get mutable references to `N` values in the map at once.
    ///
    /// Returns an array of length `N` with the results of each query. For soundness,
    /// at most one mutable reference will be returned to any value. An
    /// `Err(UnavailableMutError::Duplicate(i))` in the returned array indicates that a suitable
    /// key-value pair exists, but a mutable reference to the value already occurs at index `i` in
    /// the returned array.
    ///
    /// This method is available only if the `nightly` feature is enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::{HashMap, UnavailableMutError};
    ///
    /// let mut libraries = HashMap::new();
    /// libraries.insert("Bodleian Library".to_string(), 1602);
    /// libraries.insert("Athenæum".to_string(), 1807);
    /// libraries.insert("Herzogin-Anna-Amalia-Bibliothek".to_string(), 1691);
    /// libraries.insert("Library of Congress".to_string(), 1800);
    ///
    /// let got = libraries.get_each_mut([
    ///     "Athenæum",
    ///     "New York Public Library",
    ///     "Athenæum",
    ///     "Library of Congress",
    /// ]);
    /// assert_eq!(
    ///     got,
    ///     [
    ///         Ok(&mut 1807),
    ///         Err(UnavailableMutError::Absent),
    ///         Err(UnavailableMutError::Duplicate(0)),
    ///         Ok(&mut 1800),
    ///     ]
    /// );
    /// ```
    #[cfg(feature = "nightly")]
    pub fn get_each_mut<Q: ?Sized, const N: usize>(
        &mut self,
        ks: [&Q; N],
    ) -> [Result<&'_ mut V, UnavailableMutError>; N]
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let mut pairs = self.get_each_inner_mut(ks);
        // TODO use `MaybeUninit::uninit_array` here instead once that's stable.
        let mut out: [MaybeUninit<Result<&'_ mut V, UnavailableMutError>>; N] =
            unsafe { MaybeUninit::uninit().assume_init() };
        for i in 0..N {
            out[i] = MaybeUninit::new(
                mem::replace(&mut pairs[i], Err(UnavailableMutError::Absent)).map(|(_, v)| v),
            );
        }
        unsafe { MaybeUninit::array_assume_init(out) }
    }

    /// Attempts to get mutable references to `N` values in the map at once, with immutable
    /// references to the corresponding keys.
    ///
    /// Returns an array of length `N` with the results of each query. For soundness,
    /// at most one mutable reference will be returned to any value. An
    /// `Err(UnavailableMutError::Duplicate(i))` in the returned array indicates that a suitable
    /// key-value pair exists, but a mutable reference to the value already occurs at index `i` in
    /// the returned array.
    ///
    /// This method is available only if the `nightly` feature is enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::{HashMap, UnavailableMutError};
    ///
    /// let mut libraries = HashMap::new();
    /// libraries.insert("Bodleian Library".to_string(), 1602);
    /// libraries.insert("Athenæum".to_string(), 1807);
    /// libraries.insert("Herzogin-Anna-Amalia-Bibliothek".to_string(), 1691);
    /// libraries.insert("Library of Congress".to_string(), 1800);
    ///
    /// let got = libraries.get_each_key_value_mut([
    ///     "Bodleian Library",
    ///     "Herzogin-Anna-Amalia-Bibliothek",
    ///     "Herzogin-Anna-Amalia-Bibliothek",
    ///     "Gewandhaus",
    /// ]);
    /// assert_eq!(
    ///     got,
    ///     [
    ///         Ok((&"Bodleian Library".to_string(), &mut 1602)),
    ///         Ok((&"Herzogin-Anna-Amalia-Bibliothek".to_string(), &mut 1691)),
    ///         Err(UnavailableMutError::Duplicate(1)),
    ///         Err(UnavailableMutError::Absent),
    ///     ]
    /// );
    /// ```
    #[cfg(feature = "nightly")]
    pub fn get_each_key_value_mut<Q: ?Sized, const N: usize>(
        &mut self,
        ks: [&Q; N],
    ) -> [Result<(&'_ K, &'_ mut V), UnavailableMutError>; N]
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let mut pairs = self.get_each_inner_mut(ks);
        // TODO use `MaybeUninit::uninit_array` here instead once that's stable.
        let mut out: [MaybeUninit<Result<(&'_ K, &'_ mut V), UnavailableMutError>>; N] =
            unsafe { MaybeUninit::uninit().assume_init() };
        for i in 0..N {
            out[i] = MaybeUninit::new(
                mem::replace(&mut pairs[i], Err(UnavailableMutError::Absent))
                    .map(|(k, v)| (&*k, v)),
            );
        }
        unsafe { MaybeUninit::array_assume_init(out) }
    }

    #[cfg(feature = "nightly")]
    fn get_each_inner_mut<Q: ?Sized, const N: usize>(
        &mut self,
        ks: [&Q; N],
    ) -> [Result<&'_ mut (K, V), UnavailableMutError>; N]
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let mut hashes = [0_u64; N];
        for i in 0..N {
            hashes[i] = make_hash::<K, Q, S>(&self.hash_builder, ks[i]);
        }
        self.table
            .get_each_mut(hashes, |i, (k, _)| ks[i].eq(k.borrow()))
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, [`None`] is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old
    /// value is returned. The key is not updated, though; this matters for
    /// types that can be `==` without being identical. See the [module-level
    /// documentation] for more.
    ///
    /// [`None`]: https://doc.rust-lang.org/std/option/enum.Option.html#variant.None
    /// [module-level documentation]: index.html#insert-and-complex-keys
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// assert_eq!(map.insert(37, "a"), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert(37, "b");
    /// assert_eq!(map.insert(37, "c"), Some("b"));
    /// assert_eq!(map[&37], "c");
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        let hash = make_insert_hash::<K, S>(&self.hash_builder, &k);
        if let Some((_, item)) = self.table.get_mut(hash, equivalent_key(&k)) {
            Some(mem::replace(item, v))
        } else {
            self.table
                .insert(hash, (k, v), make_hasher::<K, _, V, S>(&self.hash_builder));
            None
        }
    }

    /// Tries to insert a key-value pair into the map, and returns
    /// a mutable reference to the value in the entry.
    ///
    /// # Errors
    ///
    /// If the map already had this key present, nothing is updated, and
    /// an error containing the occupied entry and the value is returned.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// assert_eq!(map.try_insert(37, "a").unwrap(), &"a");
    ///
    /// let err = map.try_insert(37, "b").unwrap_err();
    /// assert_eq!(err.entry.key(), &37);
    /// assert_eq!(err.entry.get(), &"a");
    /// assert_eq!(err.value, "b");
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn try_insert(
        &mut self,
        key: K,
        value: V,
    ) -> Result<&mut V, OccupiedError<'_, K, V, S, A>> {
        match self.entry(key) {
            Entry::Occupied(entry) => Err(OccupiedError { entry, value }),
            Entry::Vacant(entry) => Ok(entry.insert(value)),
        }
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
    /// [`Hash`]: https://doc.rust-lang.org/std/hash/trait.Hash.html
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove(&1), Some("a"));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn remove<Q: ?Sized>(&mut self, k: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.remove_entry(k) {
            Some((_, v)) => Some(v),
            None => None,
        }
    }

    /// Removes a key from the map, returning the stored key and value if the
    /// key was previously in the map.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
    /// [`Hash`]: https://doc.rust-lang.org/std/hash/trait.Hash.html
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove_entry(&1), Some((1, "a")));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn remove_entry<Q: ?Sized>(&mut self, k: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let hash = make_hash::<K, Q, S>(&self.hash_builder, k);
        self.table.remove_entry(hash, equivalent_key(k))
    }
}

impl<K, V, S, A: Allocator + Clone> HashMap<K, V, S, A> {
    /// Creates a raw entry builder for the HashMap.
    ///
    /// Raw entries provide the lowest level of control for searching and
    /// manipulating a map. They must be manually initialized with a hash and
    /// then manually searched. After this, insertions into a vacant entry
    /// still require an owned key to be provided.
    ///
    /// Raw entries are useful for such exotic situations as:
    ///
    /// * Hash memoization
    /// * Deferring the creation of an owned key until it is known to be required
    /// * Using a search key that doesn't work with the Borrow trait
    /// * Using custom comparison logic without newtype wrappers
    ///
    /// Because raw entries provide much more low-level control, it's much easier
    /// to put the HashMap into an inconsistent state which, while memory-safe,
    /// will cause the map to produce seemingly random results. Higher-level and
    /// more foolproof APIs like `entry` should be preferred when possible.
    ///
    /// In particular, the hash used to initialized the raw entry must still be
    /// consistent with the hash of the key that is ultimately stored in the entry.
    /// This is because implementations of HashMap may need to recompute hashes
    /// when resizing, at which point only the keys are available.
    ///
    /// Raw entries give mutable access to the keys. This must not be used
    /// to modify how the key would compare or hash, as the map will not re-evaluate
    /// where the key should go, meaning the keys may become "lost" if their
    /// location does not reflect their state. For instance, if you change a key
    /// so that the map now contains keys which compare equal, search may start
    /// acting erratically, with two keys randomly masking each other. Implementations
    /// are free to assume this doesn't happen (within the limits of memory-safety).
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn raw_entry_mut(&mut self) -> RawEntryBuilderMut<'_, K, V, S, A> {
        RawEntryBuilderMut { map: self }
    }

    /// Creates a raw immutable entry builder for the HashMap.
    ///
    /// Raw entries provide the lowest level of control for searching and
    /// manipulating a map. They must be manually initialized with a hash and
    /// then manually searched.
    ///
    /// This is useful for
    /// * Hash memoization
    /// * Using a search key that doesn't work with the Borrow trait
    /// * Using custom comparison logic without newtype wrappers
    ///
    /// Unless you are in such a situation, higher-level and more foolproof APIs like
    /// `get` should be preferred.
    ///
    /// Immutable raw entries have very limited use; you might instead want `raw_entry_mut`.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn raw_entry(&self) -> RawEntryBuilder<'_, K, V, S, A> {
        RawEntryBuilder { map: self }
    }
}

impl<K, V, S, A> PartialEq for HashMap<K, V, S, A>
where
    K: Eq + Hash,
    V: PartialEq,
    S: BuildHasher,
    A: Allocator + Clone,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        self.iter()
            .all(|(key, value)| other.get(key).map_or(false, |v| *value == *v))
    }
}

impl<K, V, S, A> Eq for HashMap<K, V, S, A>
where
    K: Eq + Hash,
    V: Eq,
    S: BuildHasher,
    A: Allocator + Clone,
{
}

impl<K, V, S, A> Debug for HashMap<K, V, S, A>
where
    K: Debug,
    V: Debug,
    A: Allocator + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V, S, A> Default for HashMap<K, V, S, A>
where
    S: Default,
    A: Default + Allocator + Clone,
{
    /// Creates an empty `HashMap<K, V, S, A>`, with the `Default` value for the hasher and allocator.
    #[cfg_attr(feature = "inline-more", inline)]
    fn default() -> Self {
        Self::with_hasher_in(Default::default(), Default::default())
    }
}

impl<K, Q: ?Sized, V, S, A> Index<&Q> for HashMap<K, V, S, A>
where
    K: Eq + Hash + Borrow<Q>,
    Q: Eq + Hash,
    S: BuildHasher,
    A: Allocator + Clone,
{
    type Output = V;

    /// Returns a reference to the value corresponding to the supplied key.
    ///
    /// # Panics
    ///
    /// Panics if the key is not present in the `HashMap`.
    #[cfg_attr(feature = "inline-more", inline)]
    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

/// An iterator over the entries of a `HashMap`.
///
/// This `struct` is created by the [`iter`] method on [`HashMap`]. See its
/// documentation for more.
///
/// [`iter`]: struct.HashMap.html#method.iter
/// [`HashMap`]: struct.HashMap.html
pub struct Iter<'a, K, V> {
    inner: RawIter<(K, V)>,
    marker: PhantomData<(&'a K, &'a V)>,
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
impl<K, V> Clone for Iter<'_, K, V> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn clone(&self) -> Self {
        Iter {
            inner: self.inner.clone(),
            marker: PhantomData,
        }
    }
}

impl<K: Debug, V: Debug> fmt::Debug for Iter<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

/// A mutable iterator over the entries of a `HashMap`.
///
/// This `struct` is created by the [`iter_mut`] method on [`HashMap`]. See its
/// documentation for more.
///
/// [`iter_mut`]: struct.HashMap.html#method.iter_mut
/// [`HashMap`]: struct.HashMap.html
pub struct IterMut<'a, K, V> {
    inner: RawIter<(K, V)>,
    // To ensure invariance with respect to V
    marker: PhantomData<(&'a K, &'a mut V)>,
}

// We override the default Send impl which has K: Sync instead of K: Send. Both
// are correct, but this one is more general since it allows keys which
// implement Send but not Sync.
unsafe impl<K: Send, V: Send> Send for IterMut<'_, K, V> {}

impl<K, V> IterMut<'_, K, V> {
    /// Returns a iterator of references over the remaining items.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(super) fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            inner: self.inner.clone(),
            marker: PhantomData,
        }
    }
}

/// An owning iterator over the entries of a `HashMap`.
///
/// This `struct` is created by the [`into_iter`] method on [`HashMap`]
/// (provided by the `IntoIterator` trait). See its documentation for more.
///
/// [`into_iter`]: struct.HashMap.html#method.into_iter
/// [`HashMap`]: struct.HashMap.html
pub struct IntoIter<K, V, A: Allocator + Clone = Global> {
    inner: RawIntoIter<(K, V), A>,
}

impl<K, V, A: Allocator + Clone> IntoIter<K, V, A> {
    /// Returns a iterator of references over the remaining items.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(super) fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            inner: self.inner.iter(),
            marker: PhantomData,
        }
    }
}

/// An iterator over the keys of a `HashMap`.
///
/// This `struct` is created by the [`keys`] method on [`HashMap`]. See its
/// documentation for more.
///
/// [`keys`]: struct.HashMap.html#method.keys
/// [`HashMap`]: struct.HashMap.html
pub struct Keys<'a, K, V> {
    inner: Iter<'a, K, V>,
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
impl<K, V> Clone for Keys<'_, K, V> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn clone(&self) -> Self {
        Keys {
            inner: self.inner.clone(),
        }
    }
}

impl<K: Debug, V> fmt::Debug for Keys<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

/// An iterator over the values of a `HashMap`.
///
/// This `struct` is created by the [`values`] method on [`HashMap`]. See its
/// documentation for more.
///
/// [`values`]: struct.HashMap.html#method.values
/// [`HashMap`]: struct.HashMap.html
pub struct Values<'a, K, V> {
    inner: Iter<'a, K, V>,
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
impl<K, V> Clone for Values<'_, K, V> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn clone(&self) -> Self {
        Values {
            inner: self.inner.clone(),
        }
    }
}

impl<K, V: Debug> fmt::Debug for Values<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

/// A draining iterator over the entries of a `HashMap`.
///
/// This `struct` is created by the [`drain`] method on [`HashMap`]. See its
/// documentation for more.
///
/// [`drain`]: struct.HashMap.html#method.drain
/// [`HashMap`]: struct.HashMap.html
pub struct Drain<'a, K, V, A: Allocator + Clone = Global> {
    inner: RawDrain<'a, (K, V), A>,
}

impl<K, V, A: Allocator + Clone> Drain<'_, K, V, A> {
    /// Returns a iterator of references over the remaining items.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(super) fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            inner: self.inner.iter(),
            marker: PhantomData,
        }
    }
}

/// A draining iterator over entries of a `HashMap` which don't satisfy the predicate `f`.
///
/// This `struct` is created by the [`drain_filter`] method on [`HashMap`]. See its
/// documentation for more.
///
/// [`drain_filter`]: struct.HashMap.html#method.drain_filter
/// [`HashMap`]: struct.HashMap.html
pub struct DrainFilter<'a, K, V, F, A: Allocator + Clone = Global>
where
    F: FnMut(&K, &mut V) -> bool,
{
    f: F,
    inner: DrainFilterInner<'a, K, V, A>,
}

impl<'a, K, V, F, A> Drop for DrainFilter<'a, K, V, F, A>
where
    F: FnMut(&K, &mut V) -> bool,
    A: Allocator + Clone,
{
    #[cfg_attr(feature = "inline-more", inline)]
    fn drop(&mut self) {
        while let Some(item) = self.next() {
            let guard = ConsumeAllOnDrop(self);
            drop(item);
            mem::forget(guard);
        }
    }
}

pub(super) struct ConsumeAllOnDrop<'a, T: Iterator>(pub &'a mut T);

impl<T: Iterator> Drop for ConsumeAllOnDrop<'_, T> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn drop(&mut self) {
        self.0.for_each(drop)
    }
}

impl<K, V, F, A> Iterator for DrainFilter<'_, K, V, F, A>
where
    F: FnMut(&K, &mut V) -> bool,
    A: Allocator + Clone,
{
    type Item = (K, V);

    #[cfg_attr(feature = "inline-more", inline)]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next(&mut self.f)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.inner.iter.size_hint().1)
    }
}

impl<K, V, F> FusedIterator for DrainFilter<'_, K, V, F> where F: FnMut(&K, &mut V) -> bool {}

/// Portions of `DrainFilter` shared with `set::DrainFilter`
pub(super) struct DrainFilterInner<'a, K, V, A: Allocator + Clone> {
    pub iter: RawIter<(K, V)>,
    pub table: &'a mut RawTable<(K, V), A>,
}

impl<K, V, A: Allocator + Clone> DrainFilterInner<'_, K, V, A> {
    #[cfg_attr(feature = "inline-more", inline)]
    pub(super) fn next<F>(&mut self, f: &mut F) -> Option<(K, V)>
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        unsafe {
            while let Some(item) = self.iter.next() {
                let &mut (ref key, ref mut value) = item.as_mut();
                if f(key, value) {
                    return Some(self.table.remove(item));
                }
            }
        }
        None
    }
}

/// A mutable iterator over the values of a `HashMap`.
///
/// This `struct` is created by the [`values_mut`] method on [`HashMap`]. See its
/// documentation for more.
///
/// [`values_mut`]: struct.HashMap.html#method.values_mut
/// [`HashMap`]: struct.HashMap.html
pub struct ValuesMut<'a, K, V> {
    inner: IterMut<'a, K, V>,
}

/// A builder for computing where in a [`HashMap`] a key-value pair would be stored.
///
/// See the [`HashMap::raw_entry_mut`] docs for usage examples.
///
/// [`HashMap::raw_entry_mut`]: struct.HashMap.html#method.raw_entry_mut
pub struct RawEntryBuilderMut<'a, K, V, S, A: Allocator + Clone = Global> {
    map: &'a mut HashMap<K, V, S, A>,
}

/// A view into a single entry in a map, which may either be vacant or occupied.
///
/// This is a lower-level version of [`Entry`].
///
/// This `enum` is constructed through the [`raw_entry_mut`] method on [`HashMap`],
/// then calling one of the methods of that [`RawEntryBuilderMut`].
///
/// [`HashMap`]: struct.HashMap.html
/// [`Entry`]: enum.Entry.html
/// [`raw_entry_mut`]: struct.HashMap.html#method.raw_entry_mut
/// [`RawEntryBuilderMut`]: struct.RawEntryBuilderMut.html
pub enum RawEntryMut<'a, K, V, S, A: Allocator + Clone = Global> {
    /// An occupied entry.
    Occupied(RawOccupiedEntryMut<'a, K, V, S, A>),
    /// A vacant entry.
    Vacant(RawVacantEntryMut<'a, K, V, S, A>),
}

/// A view into an occupied entry in a `HashMap`.
/// It is part of the [`RawEntryMut`] enum.
///
/// [`RawEntryMut`]: enum.RawEntryMut.html
pub struct RawOccupiedEntryMut<'a, K, V, S, A: Allocator + Clone = Global> {
    elem: Bucket<(K, V)>,
    table: &'a mut RawTable<(K, V), A>,
    hash_builder: &'a S,
}

unsafe impl<K, V, S, A> Send for RawOccupiedEntryMut<'_, K, V, S, A>
where
    K: Send,
    V: Send,
    A: Send + Allocator + Clone,
{
}
unsafe impl<K, V, S, A> Sync for RawOccupiedEntryMut<'_, K, V, S, A>
where
    K: Sync,
    V: Sync,
    A: Send + Allocator + Clone,
{
}

/// A view into a vacant entry in a `HashMap`.
/// It is part of the [`RawEntryMut`] enum.
///
/// [`RawEntryMut`]: enum.RawEntryMut.html
pub struct RawVacantEntryMut<'a, K, V, S, A: Allocator + Clone = Global> {
    table: &'a mut RawTable<(K, V), A>,
    hash_builder: &'a S,
}

/// A builder for computing where in a [`HashMap`] a key-value pair would be stored.
///
/// See the [`HashMap::raw_entry`] docs for usage examples.
///
/// [`HashMap::raw_entry`]: struct.HashMap.html#method.raw_entry
pub struct RawEntryBuilder<'a, K, V, S, A: Allocator + Clone = Global> {
    map: &'a HashMap<K, V, S, A>,
}

impl<'a, K, V, S, A: Allocator + Clone> RawEntryBuilderMut<'a, K, V, S, A> {
    /// Creates a `RawEntryMut` from the given key.
    #[cfg_attr(feature = "inline-more", inline)]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_key<Q: ?Sized>(self, k: &Q) -> RawEntryMut<'a, K, V, S, A>
    where
        S: BuildHasher,
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let hash = make_hash::<K, Q, S>(&self.map.hash_builder, k);
        self.from_key_hashed_nocheck(hash, k)
    }

    /// Creates a `RawEntryMut` from the given key and its hash.
    #[inline]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_key_hashed_nocheck<Q: ?Sized>(self, hash: u64, k: &Q) -> RawEntryMut<'a, K, V, S, A>
    where
        K: Borrow<Q>,
        Q: Eq,
    {
        self.from_hash(hash, equivalent(k))
    }
}

impl<'a, K, V, S, A: Allocator + Clone> RawEntryBuilderMut<'a, K, V, S, A> {
    /// Creates a `RawEntryMut` from the given hash.
    #[cfg_attr(feature = "inline-more", inline)]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_hash<F>(self, hash: u64, is_match: F) -> RawEntryMut<'a, K, V, S, A>
    where
        for<'b> F: FnMut(&'b K) -> bool,
    {
        self.search(hash, is_match)
    }

    #[cfg_attr(feature = "inline-more", inline)]
    fn search<F>(self, hash: u64, mut is_match: F) -> RawEntryMut<'a, K, V, S, A>
    where
        for<'b> F: FnMut(&'b K) -> bool,
    {
        match self.map.table.find(hash, |(k, _)| is_match(k)) {
            Some(elem) => RawEntryMut::Occupied(RawOccupiedEntryMut {
                elem,
                table: &mut self.map.table,
                hash_builder: &self.map.hash_builder,
            }),
            None => RawEntryMut::Vacant(RawVacantEntryMut {
                table: &mut self.map.table,
                hash_builder: &self.map.hash_builder,
            }),
        }
    }
}

impl<'a, K, V, S, A: Allocator + Clone> RawEntryBuilder<'a, K, V, S, A> {
    /// Access an entry by key.
    #[cfg_attr(feature = "inline-more", inline)]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_key<Q: ?Sized>(self, k: &Q) -> Option<(&'a K, &'a V)>
    where
        S: BuildHasher,
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let hash = make_hash::<K, Q, S>(&self.map.hash_builder, k);
        self.from_key_hashed_nocheck(hash, k)
    }

    /// Access an entry by a key and its hash.
    #[cfg_attr(feature = "inline-more", inline)]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_key_hashed_nocheck<Q: ?Sized>(self, hash: u64, k: &Q) -> Option<(&'a K, &'a V)>
    where
        K: Borrow<Q>,
        Q: Eq,
    {
        self.from_hash(hash, equivalent(k))
    }

    #[cfg_attr(feature = "inline-more", inline)]
    fn search<F>(self, hash: u64, mut is_match: F) -> Option<(&'a K, &'a V)>
    where
        F: FnMut(&K) -> bool,
    {
        match self.map.table.get(hash, |(k, _)| is_match(k)) {
            Some(&(ref key, ref value)) => Some((key, value)),
            None => None,
        }
    }

    /// Access an entry by hash.
    #[cfg_attr(feature = "inline-more", inline)]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_hash<F>(self, hash: u64, is_match: F) -> Option<(&'a K, &'a V)>
    where
        F: FnMut(&K) -> bool,
    {
        self.search(hash, is_match)
    }
}

impl<'a, K, V, S, A: Allocator + Clone> RawEntryMut<'a, K, V, S, A> {
    /// Sets the value of the entry, and returns a RawOccupiedEntryMut.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    /// let entry = map.raw_entry_mut().from_key("horseyland").insert("horseyland", 37);
    ///
    /// assert_eq!(entry.remove_entry(), ("horseyland", 37));
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn insert(self, key: K, value: V) -> RawOccupiedEntryMut<'a, K, V, S, A>
    where
        K: Hash,
        S: BuildHasher,
    {
        match self {
            RawEntryMut::Occupied(mut entry) => {
                entry.insert(value);
                entry
            }
            RawEntryMut::Vacant(entry) => entry.insert_entry(key, value),
        }
    }

    /// Ensures a value is in the entry by inserting the default if empty, and returns
    /// mutable references to the key and value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    ///
    /// map.raw_entry_mut().from_key("poneyland").or_insert("poneyland", 3);
    /// assert_eq!(map["poneyland"], 3);
    ///
    /// *map.raw_entry_mut().from_key("poneyland").or_insert("poneyland", 10).1 *= 2;
    /// assert_eq!(map["poneyland"], 6);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn or_insert(self, default_key: K, default_val: V) -> (&'a mut K, &'a mut V)
    where
        K: Hash,
        S: BuildHasher,
    {
        match self {
            RawEntryMut::Occupied(entry) => entry.into_key_value(),
            RawEntryMut::Vacant(entry) => entry.insert(default_key, default_val),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the default function if empty,
    /// and returns mutable references to the key and value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map: HashMap<&str, String> = HashMap::new();
    ///
    /// map.raw_entry_mut().from_key("poneyland").or_insert_with(|| {
    ///     ("poneyland", "hoho".to_string())
    /// });
    ///
    /// assert_eq!(map["poneyland"], "hoho".to_string());
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn or_insert_with<F>(self, default: F) -> (&'a mut K, &'a mut V)
    where
        F: FnOnce() -> (K, V),
        K: Hash,
        S: BuildHasher,
    {
        match self {
            RawEntryMut::Occupied(entry) => entry.into_key_value(),
            RawEntryMut::Vacant(entry) => {
                let (k, v) = default();
                entry.insert(k, v)
            }
        }
    }

    /// Provides in-place mutable access to an occupied entry before any
    /// potential inserts into the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    ///
    /// map.raw_entry_mut()
    ///    .from_key("poneyland")
    ///    .and_modify(|_k, v| { *v += 1 })
    ///    .or_insert("poneyland", 42);
    /// assert_eq!(map["poneyland"], 42);
    ///
    /// map.raw_entry_mut()
    ///    .from_key("poneyland")
    ///    .and_modify(|_k, v| { *v += 1 })
    ///    .or_insert("poneyland", 0);
    /// assert_eq!(map["poneyland"], 43);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&mut K, &mut V),
    {
        match self {
            RawEntryMut::Occupied(mut entry) => {
                {
                    let (k, v) = entry.get_key_value_mut();
                    f(k, v);
                }
                RawEntryMut::Occupied(entry)
            }
            RawEntryMut::Vacant(entry) => RawEntryMut::Vacant(entry),
        }
    }

    /// Provides shared access to the key and owned access to the value of
    /// an occupied entry and allows to replace or remove it based on the
    /// value of the returned option.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// use hashbrown::hash_map::RawEntryMut;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    ///
    /// let entry = map
    ///     .raw_entry_mut()
    ///     .from_key("poneyland")
    ///     .and_replace_entry_with(|_k, _v| panic!());
    ///
    /// match entry {
    ///     RawEntryMut::Vacant(_) => {},
    ///     RawEntryMut::Occupied(_) => panic!(),
    /// }
    ///
    /// map.insert("poneyland", 42);
    ///
    /// let entry = map
    ///     .raw_entry_mut()
    ///     .from_key("poneyland")
    ///     .and_replace_entry_with(|k, v| {
    ///         assert_eq!(k, &"poneyland");
    ///         assert_eq!(v, 42);
    ///         Some(v + 1)
    ///     });
    ///
    /// match entry {
    ///     RawEntryMut::Occupied(e) => {
    ///         assert_eq!(e.key(), &"poneyland");
    ///         assert_eq!(e.get(), &43);
    ///     },
    ///     RawEntryMut::Vacant(_) => panic!(),
    /// }
    ///
    /// assert_eq!(map["poneyland"], 43);
    ///
    /// let entry = map
    ///     .raw_entry_mut()
    ///     .from_key("poneyland")
    ///     .and_replace_entry_with(|_k, _v| None);
    ///
    /// match entry {
    ///     RawEntryMut::Vacant(_) => {},
    ///     RawEntryMut::Occupied(_) => panic!(),
    /// }
    ///
    /// assert!(!map.contains_key("poneyland"));
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn and_replace_entry_with<F>(self, f: F) -> Self
    where
        F: FnOnce(&K, V) -> Option<V>,
    {
        match self {
            RawEntryMut::Occupied(entry) => entry.replace_entry_with(f),
            RawEntryMut::Vacant(_) => self,
        }
    }
}

impl<'a, K, V, S, A: Allocator + Clone> RawOccupiedEntryMut<'a, K, V, S, A> {
    /// Gets a reference to the key in the entry.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn key(&self) -> &K {
        unsafe { &self.elem.as_ref().0 }
    }

    /// Gets a mutable reference to the key in the entry.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn key_mut(&mut self) -> &mut K {
        unsafe { &mut self.elem.as_mut().0 }
    }

    /// Converts the entry into a mutable reference to the key in the entry
    /// with a lifetime bound to the map itself.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn into_key(self) -> &'a mut K {
        unsafe { &mut self.elem.as_mut().0 }
    }

    /// Gets a reference to the value in the entry.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn get(&self) -> &V {
        unsafe { &self.elem.as_ref().1 }
    }

    /// Converts the OccupiedEntry into a mutable reference to the value in the entry
    /// with a lifetime bound to the map itself.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn into_mut(self) -> &'a mut V {
        unsafe { &mut self.elem.as_mut().1 }
    }

    /// Gets a mutable reference to the value in the entry.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn get_mut(&mut self) -> &mut V {
        unsafe { &mut self.elem.as_mut().1 }
    }

    /// Gets a reference to the key and value in the entry.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn get_key_value(&mut self) -> (&K, &V) {
        unsafe {
            let &(ref key, ref value) = self.elem.as_ref();
            (key, value)
        }
    }

    /// Gets a mutable reference to the key and value in the entry.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn get_key_value_mut(&mut self) -> (&mut K, &mut V) {
        unsafe {
            let &mut (ref mut key, ref mut value) = self.elem.as_mut();
            (key, value)
        }
    }

    /// Converts the OccupiedEntry into a mutable reference to the key and value in the entry
    /// with a lifetime bound to the map itself.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn into_key_value(self) -> (&'a mut K, &'a mut V) {
        unsafe {
            let &mut (ref mut key, ref mut value) = self.elem.as_mut();
            (key, value)
        }
    }

    /// Sets the value of the entry, and returns the entry's old value.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn insert(&mut self, value: V) -> V {
        mem::replace(self.get_mut(), value)
    }

    /// Sets the value of the entry, and returns the entry's old value.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn insert_key(&mut self, key: K) -> K {
        mem::replace(self.key_mut(), key)
    }

    /// Takes the value out of the entry, and returns it.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    /// Take the ownership of the key and value from the map.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn remove_entry(self) -> (K, V) {
        unsafe { self.table.remove(self.elem) }
    }

    /// Provides shared access to the key and owned access to the value of
    /// the entry and allows to replace or remove it based on the
    /// value of the returned option.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn replace_entry_with<F>(self, f: F) -> RawEntryMut<'a, K, V, S, A>
    where
        F: FnOnce(&K, V) -> Option<V>,
    {
        unsafe {
            let still_occupied = self
                .table
                .replace_bucket_with(self.elem.clone(), |(key, value)| {
                    f(&key, value).map(|new_value| (key, new_value))
                });

            if still_occupied {
                RawEntryMut::Occupied(self)
            } else {
                RawEntryMut::Vacant(RawVacantEntryMut {
                    table: self.table,
                    hash_builder: self.hash_builder,
                })
            }
        }
    }
}

impl<'a, K, V, S, A: Allocator + Clone> RawVacantEntryMut<'a, K, V, S, A> {
    /// Sets the value of the entry with the VacantEntry's key,
    /// and returns a mutable reference to it.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn insert(self, key: K, value: V) -> (&'a mut K, &'a mut V)
    where
        K: Hash,
        S: BuildHasher,
    {
        let hash = make_insert_hash::<K, S>(self.hash_builder, &key);
        self.insert_hashed_nocheck(hash, key, value)
    }

    /// Sets the value of the entry with the VacantEntry's key,
    /// and returns a mutable reference to it.
    #[cfg_attr(feature = "inline-more", inline)]
    #[allow(clippy::shadow_unrelated)]
    pub fn insert_hashed_nocheck(self, hash: u64, key: K, value: V) -> (&'a mut K, &'a mut V)
    where
        K: Hash,
        S: BuildHasher,
    {
        let &mut (ref mut k, ref mut v) = self.table.insert_entry(
            hash,
            (key, value),
            make_hasher::<K, _, V, S>(self.hash_builder),
        );
        (k, v)
    }

    /// Set the value of an entry with a custom hasher function.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn insert_with_hasher<H>(
        self,
        hash: u64,
        key: K,
        value: V,
        hasher: H,
    ) -> (&'a mut K, &'a mut V)
    where
        H: Fn(&K) -> u64,
    {
        let &mut (ref mut k, ref mut v) = self
            .table
            .insert_entry(hash, (key, value), |x| hasher(&x.0));
        (k, v)
    }

    #[cfg_attr(feature = "inline-more", inline)]
    fn insert_entry(self, key: K, value: V) -> RawOccupiedEntryMut<'a, K, V, S, A>
    where
        K: Hash,
        S: BuildHasher,
    {
        let hash = make_insert_hash::<K, S>(self.hash_builder, &key);
        let elem = self.table.insert(
            hash,
            (key, value),
            make_hasher::<K, _, V, S>(self.hash_builder),
        );
        RawOccupiedEntryMut {
            elem,
            table: self.table,
            hash_builder: self.hash_builder,
        }
    }
}

impl<K, V, S, A: Allocator + Clone> Debug for RawEntryBuilderMut<'_, K, V, S, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawEntryBuilder").finish()
    }
}

impl<K: Debug, V: Debug, S, A: Allocator + Clone> Debug for RawEntryMut<'_, K, V, S, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            RawEntryMut::Vacant(ref v) => f.debug_tuple("RawEntry").field(v).finish(),
            RawEntryMut::Occupied(ref o) => f.debug_tuple("RawEntry").field(o).finish(),
        }
    }
}

impl<K: Debug, V: Debug, S, A: Allocator + Clone> Debug for RawOccupiedEntryMut<'_, K, V, S, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawOccupiedEntryMut")
            .field("key", self.key())
            .field("value", self.get())
            .finish()
    }
}

impl<K, V, S, A: Allocator + Clone> Debug for RawVacantEntryMut<'_, K, V, S, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawVacantEntryMut").finish()
    }
}

impl<K, V, S, A: Allocator + Clone> Debug for RawEntryBuilder<'_, K, V, S, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawEntryBuilder").finish()
    }
}

/// A view into a single entry in a map, which may either be vacant or occupied.
///
/// This `enum` is constructed from the [`entry`] method on [`HashMap`].
///
/// [`HashMap`]: struct.HashMap.html
/// [`entry`]: struct.HashMap.html#method.entry
pub enum Entry<'a, K, V, S, A = Global>
where
    A: Allocator + Clone,
{
    /// An occupied entry.
    Occupied(OccupiedEntry<'a, K, V, S, A>),

    /// A vacant entry.
    Vacant(VacantEntry<'a, K, V, S, A>),
}

impl<K: Debug, V: Debug, S, A: Allocator + Clone> Debug for Entry<'_, K, V, S, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Entry::Vacant(ref v) => f.debug_tuple("Entry").field(v).finish(),
            Entry::Occupied(ref o) => f.debug_tuple("Entry").field(o).finish(),
        }
    }
}

/// A view into an occupied entry in a `HashMap`.
/// It is part of the [`Entry`] enum.
///
/// [`Entry`]: enum.Entry.html
pub struct OccupiedEntry<'a, K, V, S, A: Allocator + Clone = Global> {
    hash: u64,
    key: Option<K>,
    elem: Bucket<(K, V)>,
    table: &'a mut HashMap<K, V, S, A>,
}

unsafe impl<K, V, S, A> Send for OccupiedEntry<'_, K, V, S, A>
where
    K: Send,
    V: Send,
    S: Send,
    A: Send + Allocator + Clone,
{
}
unsafe impl<K, V, S, A> Sync for OccupiedEntry<'_, K, V, S, A>
where
    K: Sync,
    V: Sync,
    S: Sync,
    A: Sync + Allocator + Clone,
{
}

impl<K: Debug, V: Debug, S, A: Allocator + Clone> Debug for OccupiedEntry<'_, K, V, S, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedEntry")
            .field("key", self.key())
            .field("value", self.get())
            .finish()
    }
}

/// A view into a vacant entry in a `HashMap`.
/// It is part of the [`Entry`] enum.
///
/// [`Entry`]: enum.Entry.html
pub struct VacantEntry<'a, K, V, S, A: Allocator + Clone = Global> {
    hash: u64,
    key: K,
    table: &'a mut HashMap<K, V, S, A>,
}

impl<K: Debug, V, S, A: Allocator + Clone> Debug for VacantEntry<'_, K, V, S, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("VacantEntry").field(self.key()).finish()
    }
}

/// The error returned by [`try_insert`](HashMap::try_insert) when the key already exists.
///
/// Contains the occupied entry, and the value that was not inserted.
pub struct OccupiedError<'a, K, V, S, A: Allocator + Clone = Global> {
    /// The entry in the map that was already occupied.
    pub entry: OccupiedEntry<'a, K, V, S, A>,
    /// The value which was not inserted, because the entry was already occupied.
    pub value: V,
}

impl<K: Debug, V: Debug, S, A: Allocator + Clone> Debug for OccupiedError<'_, K, V, S, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedError")
            .field("key", self.entry.key())
            .field("old_value", self.entry.get())
            .field("new_value", &self.value)
            .finish()
    }
}

impl<'a, K: Debug, V: Debug, S, A: Allocator + Clone> fmt::Display
    for OccupiedError<'a, K, V, S, A>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "failed to insert {:?}, key {:?} already exists with value {:?}",
            self.value,
            self.entry.key(),
            self.entry.get(),
        )
    }
}

impl<'a, K, V, S, A: Allocator + Clone> IntoIterator for &'a HashMap<K, V, S, A> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    #[cfg_attr(feature = "inline-more", inline)]
    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

impl<'a, K, V, S, A: Allocator + Clone> IntoIterator for &'a mut HashMap<K, V, S, A> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    #[cfg_attr(feature = "inline-more", inline)]
    fn into_iter(self) -> IterMut<'a, K, V> {
        self.iter_mut()
    }
}

impl<K, V, S, A: Allocator + Clone> IntoIterator for HashMap<K, V, S, A> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, A>;

    /// Creates a consuming iterator, that is, one that moves each key-value
    /// pair out of the map in arbitrary order. The map cannot be used after
    /// calling this.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// // Not possible with .iter()
    /// let vec: Vec<(&str, i32)> = map.into_iter().collect();
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    fn into_iter(self) -> IntoIter<K, V, A> {
        IntoIter {
            inner: self.table.into_iter(),
        }
    }
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    #[cfg_attr(feature = "inline-more", inline)]
    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.inner.next() {
            Some(x) => unsafe {
                let r = x.as_ref();
                Some((&r.0, &r.1))
            },
            None => None,
        }
    }
    #[cfg_attr(feature = "inline-more", inline)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}
impl<K, V> ExactSizeIterator for Iter<'_, K, V> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<K, V> FusedIterator for Iter<'_, K, V> {}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    #[cfg_attr(feature = "inline-more", inline)]
    fn next(&mut self) -> Option<(&'a K, &'a mut V)> {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.inner.next() {
            Some(x) => unsafe {
                let r = x.as_mut();
                Some((&r.0, &mut r.1))
            },
            None => None,
        }
    }
    #[cfg_attr(feature = "inline-more", inline)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}
impl<K, V> ExactSizeIterator for IterMut<'_, K, V> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn len(&self) -> usize {
        self.inner.len()
    }
}
impl<K, V> FusedIterator for IterMut<'_, K, V> {}

impl<K, V> fmt::Debug for IterMut<'_, K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<K, V, A: Allocator + Clone> Iterator for IntoIter<K, V, A> {
    type Item = (K, V);

    #[cfg_attr(feature = "inline-more", inline)]
    fn next(&mut self) -> Option<(K, V)> {
        self.inner.next()
    }
    #[cfg_attr(feature = "inline-more", inline)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}
impl<K, V, A: Allocator + Clone> ExactSizeIterator for IntoIter<K, V, A> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn len(&self) -> usize {
        self.inner.len()
    }
}
impl<K, V, A: Allocator + Clone> FusedIterator for IntoIter<K, V, A> {}

impl<K: Debug, V: Debug, A: Allocator + Clone> fmt::Debug for IntoIter<K, V, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    #[cfg_attr(feature = "inline-more", inline)]
    fn next(&mut self) -> Option<&'a K> {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.inner.next() {
            Some((k, _)) => Some(k),
            None => None,
        }
    }
    #[cfg_attr(feature = "inline-more", inline)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}
impl<K, V> ExactSizeIterator for Keys<'_, K, V> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn len(&self) -> usize {
        self.inner.len()
    }
}
impl<K, V> FusedIterator for Keys<'_, K, V> {}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    #[cfg_attr(feature = "inline-more", inline)]
    fn next(&mut self) -> Option<&'a V> {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.inner.next() {
            Some((_, v)) => Some(v),
            None => None,
        }
    }
    #[cfg_attr(feature = "inline-more", inline)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}
impl<K, V> ExactSizeIterator for Values<'_, K, V> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn len(&self) -> usize {
        self.inner.len()
    }
}
impl<K, V> FusedIterator for Values<'_, K, V> {}

impl<'a, K, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;

    #[cfg_attr(feature = "inline-more", inline)]
    fn next(&mut self) -> Option<&'a mut V> {
        // Avoid `Option::map` because it bloats LLVM IR.
        match self.inner.next() {
            Some((_, v)) => Some(v),
            None => None,
        }
    }
    #[cfg_attr(feature = "inline-more", inline)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}
impl<K, V> ExactSizeIterator for ValuesMut<'_, K, V> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn len(&self) -> usize {
        self.inner.len()
    }
}
impl<K, V> FusedIterator for ValuesMut<'_, K, V> {}

impl<K, V> fmt::Debug for ValuesMut<'_, K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.inner.iter()).finish()
    }
}

impl<'a, K, V, A: Allocator + Clone> Iterator for Drain<'a, K, V, A> {
    type Item = (K, V);

    #[cfg_attr(feature = "inline-more", inline)]
    fn next(&mut self) -> Option<(K, V)> {
        self.inner.next()
    }
    #[cfg_attr(feature = "inline-more", inline)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}
impl<K, V, A: Allocator + Clone> ExactSizeIterator for Drain<'_, K, V, A> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn len(&self) -> usize {
        self.inner.len()
    }
}
impl<K, V, A: Allocator + Clone> FusedIterator for Drain<'_, K, V, A> {}

impl<K, V, A> fmt::Debug for Drain<'_, K, V, A>
where
    K: fmt::Debug,
    V: fmt::Debug,
    A: Allocator + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, K, V, S, A: Allocator + Clone> Entry<'a, K, V, S, A> {
    /// Sets the value of the entry, and returns an OccupiedEntry.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    /// let entry = map.entry("horseyland").insert(37);
    ///
    /// assert_eq!(entry.key(), &"horseyland");
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn insert(self, value: V) -> OccupiedEntry<'a, K, V, S, A>
    where
        K: Hash,
        S: BuildHasher,
    {
        match self {
            Entry::Occupied(mut entry) => {
                entry.insert(value);
                entry
            }
            Entry::Vacant(entry) => entry.insert_entry(value),
        }
    }

    /// Ensures a value is in the entry by inserting the default if empty, and returns
    /// a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    ///
    /// map.entry("poneyland").or_insert(3);
    /// assert_eq!(map["poneyland"], 3);
    ///
    /// *map.entry("poneyland").or_insert(10) *= 2;
    /// assert_eq!(map["poneyland"], 6);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn or_insert(self, default: V) -> &'a mut V
    where
        K: Hash,
        S: BuildHasher,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the default function if empty,
    /// and returns a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map: HashMap<&str, String> = HashMap::new();
    /// let s = "hoho".to_string();
    ///
    /// map.entry("poneyland").or_insert_with(|| s);
    ///
    /// assert_eq!(map["poneyland"], "hoho".to_string());
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V
    where
        K: Hash,
        S: BuildHasher,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default()),
        }
    }

    /// Ensures a value is in the entry by inserting, if empty, the result of the default function.
    /// This method allows for generating key-derived values for insertion by providing the default
    /// function a reference to the key that was moved during the `.entry(key)` method call.
    ///
    /// The reference to the moved key is provided so that cloning or copying the key is
    /// unnecessary, unlike with `.or_insert_with(|| ... )`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map: HashMap<&str, usize> = HashMap::new();
    ///
    /// map.entry("poneyland").or_insert_with_key(|key| key.chars().count());
    ///
    /// assert_eq!(map["poneyland"], 9);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn or_insert_with_key<F: FnOnce(&K) -> V>(self, default: F) -> &'a mut V
    where
        K: Hash,
        S: BuildHasher,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let value = default(entry.key());
                entry.insert(value)
            }
        }
    }

    /// Returns a reference to this entry's key.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    /// assert_eq!(map.entry("poneyland").key(), &"poneyland");
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn key(&self) -> &K {
        match *self {
            Entry::Occupied(ref entry) => entry.key(),
            Entry::Vacant(ref entry) => entry.key(),
        }
    }

    /// Provides in-place mutable access to an occupied entry before any
    /// potential inserts into the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    ///
    /// map.entry("poneyland")
    ///    .and_modify(|e| { *e += 1 })
    ///    .or_insert(42);
    /// assert_eq!(map["poneyland"], 42);
    ///
    /// map.entry("poneyland")
    ///    .and_modify(|e| { *e += 1 })
    ///    .or_insert(42);
    /// assert_eq!(map["poneyland"], 43);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        match self {
            Entry::Occupied(mut entry) => {
                f(entry.get_mut());
                Entry::Occupied(entry)
            }
            Entry::Vacant(entry) => Entry::Vacant(entry),
        }
    }

    /// Provides shared access to the key and owned access to the value of
    /// an occupied entry and allows to replace or remove it based on the
    /// value of the returned option.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// use hashbrown::hash_map::Entry;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    ///
    /// let entry = map
    ///     .entry("poneyland")
    ///     .and_replace_entry_with(|_k, _v| panic!());
    ///
    /// match entry {
    ///     Entry::Vacant(e) => {
    ///         assert_eq!(e.key(), &"poneyland");
    ///     }
    ///     Entry::Occupied(_) => panic!(),
    /// }
    ///
    /// map.insert("poneyland", 42);
    ///
    /// let entry = map
    ///     .entry("poneyland")
    ///     .and_replace_entry_with(|k, v| {
    ///         assert_eq!(k, &"poneyland");
    ///         assert_eq!(v, 42);
    ///         Some(v + 1)
    ///     });
    ///
    /// match entry {
    ///     Entry::Occupied(e) => {
    ///         assert_eq!(e.key(), &"poneyland");
    ///         assert_eq!(e.get(), &43);
    ///     }
    ///     Entry::Vacant(_) => panic!(),
    /// }
    ///
    /// assert_eq!(map["poneyland"], 43);
    ///
    /// let entry = map
    ///     .entry("poneyland")
    ///     .and_replace_entry_with(|_k, _v| None);
    ///
    /// match entry {
    ///     Entry::Vacant(e) => assert_eq!(e.key(), &"poneyland"),
    ///     Entry::Occupied(_) => panic!(),
    /// }
    ///
    /// assert!(!map.contains_key("poneyland"));
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn and_replace_entry_with<F>(self, f: F) -> Self
    where
        F: FnOnce(&K, V) -> Option<V>,
    {
        match self {
            Entry::Occupied(entry) => entry.replace_entry_with(f),
            Entry::Vacant(_) => self,
        }
    }
}

impl<'a, K, V: Default, S, A: Allocator + Clone> Entry<'a, K, V, S, A> {
    /// Ensures a value is in the entry by inserting the default value if empty,
    /// and returns a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map: HashMap<&str, Option<u32>> = HashMap::new();
    /// map.entry("poneyland").or_default();
    ///
    /// assert_eq!(map["poneyland"], None);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn or_default(self) -> &'a mut V
    where
        K: Hash,
        S: BuildHasher,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(Default::default()),
        }
    }
}

impl<'a, K, V, S, A: Allocator + Clone> OccupiedEntry<'a, K, V, S, A> {
    /// Gets a reference to the key in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    /// map.entry("poneyland").or_insert(12);
    /// assert_eq!(map.entry("poneyland").key(), &"poneyland");
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn key(&self) -> &K {
        unsafe { &self.elem.as_ref().0 }
    }

    /// Take the ownership of the key and value from the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// use hashbrown::hash_map::Entry;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// if let Entry::Occupied(o) = map.entry("poneyland") {
    ///     // We delete the entry from the map.
    ///     o.remove_entry();
    /// }
    ///
    /// assert_eq!(map.contains_key("poneyland"), false);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn remove_entry(self) -> (K, V) {
        unsafe { self.table.table.remove(self.elem) }
    }

    /// Gets a reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// use hashbrown::hash_map::Entry;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// if let Entry::Occupied(o) = map.entry("poneyland") {
    ///     assert_eq!(o.get(), &12);
    /// }
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn get(&self) -> &V {
        unsafe { &self.elem.as_ref().1 }
    }

    /// Gets a mutable reference to the value in the entry.
    ///
    /// If you need a reference to the `OccupiedEntry` which may outlive the
    /// destruction of the `Entry` value, see [`into_mut`].
    ///
    /// [`into_mut`]: #method.into_mut
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// use hashbrown::hash_map::Entry;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// assert_eq!(map["poneyland"], 12);
    /// if let Entry::Occupied(mut o) = map.entry("poneyland") {
    ///     *o.get_mut() += 10;
    ///     assert_eq!(*o.get(), 22);
    ///
    ///     // We can use the same Entry multiple times.
    ///     *o.get_mut() += 2;
    /// }
    ///
    /// assert_eq!(map["poneyland"], 24);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn get_mut(&mut self) -> &mut V {
        unsafe { &mut self.elem.as_mut().1 }
    }

    /// Converts the OccupiedEntry into a mutable reference to the value in the entry
    /// with a lifetime bound to the map itself.
    ///
    /// If you need multiple references to the `OccupiedEntry`, see [`get_mut`].
    ///
    /// [`get_mut`]: #method.get_mut
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// use hashbrown::hash_map::Entry;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// assert_eq!(map["poneyland"], 12);
    /// if let Entry::Occupied(o) = map.entry("poneyland") {
    ///     *o.into_mut() += 10;
    /// }
    ///
    /// assert_eq!(map["poneyland"], 22);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn into_mut(self) -> &'a mut V {
        unsafe { &mut self.elem.as_mut().1 }
    }

    /// Sets the value of the entry, and returns the entry's old value.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// use hashbrown::hash_map::Entry;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// if let Entry::Occupied(mut o) = map.entry("poneyland") {
    ///     assert_eq!(o.insert(15), 12);
    /// }
    ///
    /// assert_eq!(map["poneyland"], 15);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn insert(&mut self, mut value: V) -> V {
        let old_value = self.get_mut();
        mem::swap(&mut value, old_value);
        value
    }

    /// Takes the value out of the entry, and returns it.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// use hashbrown::hash_map::Entry;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    /// map.entry("poneyland").or_insert(12);
    ///
    /// if let Entry::Occupied(o) = map.entry("poneyland") {
    ///     assert_eq!(o.remove(), 12);
    /// }
    ///
    /// assert_eq!(map.contains_key("poneyland"), false);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    /// Replaces the entry, returning the old key and value. The new key in the hash map will be
    /// the key used to create this entry.
    ///
    /// # Panics
    ///
    /// Will panic if this OccupiedEntry was created through [`Entry::insert`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::hash_map::{Entry, HashMap};
    /// use std::rc::Rc;
    ///
    /// let mut map: HashMap<Rc<String>, u32> = HashMap::new();
    /// map.insert(Rc::new("Stringthing".to_string()), 15);
    ///
    /// let my_key = Rc::new("Stringthing".to_string());
    ///
    /// if let Entry::Occupied(entry) = map.entry(my_key) {
    ///     // Also replace the key with a handle to our other key.
    ///     let (old_key, old_value): (Rc<String>, u32) = entry.replace_entry(16);
    /// }
    ///
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn replace_entry(self, value: V) -> (K, V) {
        let entry = unsafe { self.elem.as_mut() };

        let old_key = mem::replace(&mut entry.0, self.key.unwrap());
        let old_value = mem::replace(&mut entry.1, value);

        (old_key, old_value)
    }

    /// Replaces the key in the hash map with the key used to create this entry.
    ///
    /// # Panics
    ///
    /// Will panic if this OccupiedEntry was created through [`Entry::insert`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::hash_map::{Entry, HashMap};
    /// use std::rc::Rc;
    ///
    /// let mut map: HashMap<Rc<String>, u32> = HashMap::new();
    /// let mut known_strings: Vec<Rc<String>> = Vec::new();
    ///
    /// // Initialise known strings, run program, etc.
    ///
    /// reclaim_memory(&mut map, &known_strings);
    ///
    /// fn reclaim_memory(map: &mut HashMap<Rc<String>, u32>, known_strings: &[Rc<String>] ) {
    ///     for s in known_strings {
    ///         if let Entry::Occupied(entry) = map.entry(s.clone()) {
    ///             // Replaces the entry's key with our version of it in `known_strings`.
    ///             entry.replace_key();
    ///         }
    ///     }
    /// }
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn replace_key(self) -> K {
        let entry = unsafe { self.elem.as_mut() };
        mem::replace(&mut entry.0, self.key.unwrap())
    }

    /// Provides shared access to the key and owned access to the value of
    /// the entry and allows to replace or remove it based on the
    /// value of the returned option.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// use hashbrown::hash_map::Entry;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    /// map.insert("poneyland", 42);
    ///
    /// let entry = match map.entry("poneyland") {
    ///     Entry::Occupied(e) => {
    ///         e.replace_entry_with(|k, v| {
    ///             assert_eq!(k, &"poneyland");
    ///             assert_eq!(v, 42);
    ///             Some(v + 1)
    ///         })
    ///     }
    ///     Entry::Vacant(_) => panic!(),
    /// };
    ///
    /// match entry {
    ///     Entry::Occupied(e) => {
    ///         assert_eq!(e.key(), &"poneyland");
    ///         assert_eq!(e.get(), &43);
    ///     }
    ///     Entry::Vacant(_) => panic!(),
    /// }
    ///
    /// assert_eq!(map["poneyland"], 43);
    ///
    /// let entry = match map.entry("poneyland") {
    ///     Entry::Occupied(e) => e.replace_entry_with(|_k, _v| None),
    ///     Entry::Vacant(_) => panic!(),
    /// };
    ///
    /// match entry {
    ///     Entry::Vacant(e) => {
    ///         assert_eq!(e.key(), &"poneyland");
    ///     }
    ///     Entry::Occupied(_) => panic!(),
    /// }
    ///
    /// assert!(!map.contains_key("poneyland"));
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn replace_entry_with<F>(self, f: F) -> Entry<'a, K, V, S, A>
    where
        F: FnOnce(&K, V) -> Option<V>,
    {
        unsafe {
            let mut spare_key = None;

            self.table
                .table
                .replace_bucket_with(self.elem.clone(), |(key, value)| {
                    if let Some(new_value) = f(&key, value) {
                        Some((key, new_value))
                    } else {
                        spare_key = Some(key);
                        None
                    }
                });

            if let Some(key) = spare_key {
                Entry::Vacant(VacantEntry {
                    hash: self.hash,
                    key,
                    table: self.table,
                })
            } else {
                Entry::Occupied(self)
            }
        }
    }
}

impl<'a, K, V, S, A: Allocator + Clone> VacantEntry<'a, K, V, S, A> {
    /// Gets a reference to the key that would be used when inserting a value
    /// through the `VacantEntry`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    /// assert_eq!(map.entry("poneyland").key(), &"poneyland");
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Take ownership of the key.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// use hashbrown::hash_map::Entry;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    ///
    /// if let Entry::Vacant(v) = map.entry("poneyland") {
    ///     v.into_key();
    /// }
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn into_key(self) -> K {
        self.key
    }

    /// Sets the value of the entry with the VacantEntry's key,
    /// and returns a mutable reference to it.
    ///
    /// # Examples
    ///
    /// ```
    /// use hashbrown::HashMap;
    /// use hashbrown::hash_map::Entry;
    ///
    /// let mut map: HashMap<&str, u32> = HashMap::new();
    ///
    /// if let Entry::Vacant(o) = map.entry("poneyland") {
    ///     o.insert(37);
    /// }
    /// assert_eq!(map["poneyland"], 37);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn insert(self, value: V) -> &'a mut V
    where
        K: Hash,
        S: BuildHasher,
    {
        let table = &mut self.table.table;
        let entry = table.insert_entry(
            self.hash,
            (self.key, value),
            make_hasher::<K, _, V, S>(&self.table.hash_builder),
        );
        &mut entry.1
    }

    #[cfg_attr(feature = "inline-more", inline)]
    fn insert_entry(self, value: V) -> OccupiedEntry<'a, K, V, S, A>
    where
        K: Hash,
        S: BuildHasher,
    {
        let elem = self.table.table.insert(
            self.hash,
            (self.key, value),
            make_hasher::<K, _, V, S>(&self.table.hash_builder),
        );
        OccupiedEntry {
            hash: self.hash,
            key: None,
            elem,
            table: self.table,
        }
    }
}

impl<K, V, S, A> FromIterator<(K, V)> for HashMap<K, V, S, A>
where
    K: Eq + Hash,
    S: BuildHasher + Default,
    A: Default + Allocator + Clone,
{
    #[cfg_attr(feature = "inline-more", inline)]
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let mut map =
            Self::with_capacity_and_hasher_in(iter.size_hint().0, S::default(), A::default());
        iter.for_each(|(k, v)| {
            map.insert(k, v);
        });
        map
    }
}

/// Inserts all new key-values from the iterator and replaces values with existing
/// keys with new values returned from the iterator.
impl<K, V, S, A> Extend<(K, V)> for HashMap<K, V, S, A>
where
    K: Eq + Hash,
    S: BuildHasher,
    A: Allocator + Clone,
{
    #[cfg_attr(feature = "inline-more", inline)]
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        // Keys may be already present or show multiple times in the iterator.
        // Reserve the entire hint lower bound if the map is empty.
        // Otherwise reserve half the hint (rounded up), so the map
        // will only resize twice in the worst case.
        let iter = iter.into_iter();
        let reserve = if self.is_empty() {
            iter.size_hint().0
        } else {
            (iter.size_hint().0 + 1) / 2
        };
        self.reserve(reserve);
        iter.for_each(move |(k, v)| {
            self.insert(k, v);
        });
    }

    #[inline]
    #[cfg(feature = "nightly")]
    fn extend_one(&mut self, (k, v): (K, V)) {
        self.insert(k, v);
    }

    #[inline]
    #[cfg(feature = "nightly")]
    fn extend_reserve(&mut self, additional: usize) {
        // Keys may be already present or show multiple times in the iterator.
        // Reserve the entire hint lower bound if the map is empty.
        // Otherwise reserve half the hint (rounded up), so the map
        // will only resize twice in the worst case.
        let reserve = if self.is_empty() {
            additional
        } else {
            (additional + 1) / 2
        };
        self.reserve(reserve);
    }
}

impl<'a, K, V, S, A> Extend<(&'a K, &'a V)> for HashMap<K, V, S, A>
where
    K: Eq + Hash + Copy,
    V: Copy,
    S: BuildHasher,
    A: Allocator + Clone,
{
    #[cfg_attr(feature = "inline-more", inline)]
    fn extend<T: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: T) {
        self.extend(iter.into_iter().map(|(&key, &value)| (key, value)));
    }

    #[inline]
    #[cfg(feature = "nightly")]
    fn extend_one(&mut self, (k, v): (&'a K, &'a V)) {
        self.insert(*k, *v);
    }

    #[inline]
    #[cfg(feature = "nightly")]
    fn extend_reserve(&mut self, additional: usize) {
        Extend::<(K, V)>::extend_reserve(self, additional);
    }
}

#[allow(dead_code)]
fn assert_covariance() {
    fn map_key<'new>(v: HashMap<&'static str, u8>) -> HashMap<&'new str, u8> {
        v
    }
    fn map_val<'new>(v: HashMap<u8, &'static str>) -> HashMap<u8, &'new str> {
        v
    }
    fn iter_key<'a, 'new>(v: Iter<'a, &'static str, u8>) -> Iter<'a, &'new str, u8> {
        v
    }
    fn iter_val<'a, 'new>(v: Iter<'a, u8, &'static str>) -> Iter<'a, u8, &'new str> {
        v
    }
    fn into_iter_key<'new, A: Allocator + Clone>(
        v: IntoIter<&'static str, u8, A>,
    ) -> IntoIter<&'new str, u8, A> {
        v
    }
    fn into_iter_val<'new, A: Allocator + Clone>(
        v: IntoIter<u8, &'static str, A>,
    ) -> IntoIter<u8, &'new str, A> {
        v
    }
    fn keys_key<'a, 'new>(v: Keys<'a, &'static str, u8>) -> Keys<'a, &'new str, u8> {
        v
    }
    fn keys_val<'a, 'new>(v: Keys<'a, u8, &'static str>) -> Keys<'a, u8, &'new str> {
        v
    }
    fn values_key<'a, 'new>(v: Values<'a, &'static str, u8>) -> Values<'a, &'new str, u8> {
        v
    }
    fn values_val<'a, 'new>(v: Values<'a, u8, &'static str>) -> Values<'a, u8, &'new str> {
        v
    }
    fn drain<'new>(
        d: Drain<'static, &'static str, &'static str>,
    ) -> Drain<'new, &'new str, &'new str> {
        d
    }
}

#[cfg(test)]
mod test_map {
    use super::DefaultHashBuilder;
    use super::Entry::{Occupied, Vacant};
    use super::{HashMap, RawEntryMut};
    use crate::TryReserveError::*;
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use std::borrow::ToOwned;
    use std::cell::RefCell;
    use std::usize;
    use std::vec::Vec;

    #[test]
    fn test_zero_capacities() {
        type HM = HashMap<i32, i32>;

        let m = HM::new();
        assert_eq!(m.capacity(), 0);

        let m = HM::default();
        assert_eq!(m.capacity(), 0);

        let m = HM::with_hasher(DefaultHashBuilder::default());
        assert_eq!(m.capacity(), 0);

        let m = HM::with_capacity(0);
        assert_eq!(m.capacity(), 0);

        let m = HM::with_capacity_and_hasher(0, DefaultHashBuilder::default());
        assert_eq!(m.capacity(), 0);

        let mut m = HM::new();
        m.insert(1, 1);
        m.insert(2, 2);
        m.remove(&1);
        m.remove(&2);
        m.shrink_to_fit();
        assert_eq!(m.capacity(), 0);

        let mut m = HM::new();
        m.reserve(0);
        assert_eq!(m.capacity(), 0);
    }

    #[test]
    fn test_create_capacity_zero() {
        let mut m = HashMap::with_capacity(0);

        assert!(m.insert(1, 1).is_none());

        assert!(m.contains_key(&1));
        assert!(!m.contains_key(&0));
    }

    #[test]
    fn test_insert() {
        let mut m = HashMap::new();
        assert_eq!(m.len(), 0);
        assert!(m.insert(1, 2).is_none());
        assert_eq!(m.len(), 1);
        assert!(m.insert(2, 4).is_none());
        assert_eq!(m.len(), 2);
        assert_eq!(*m.get(&1).unwrap(), 2);
        assert_eq!(*m.get(&2).unwrap(), 4);
    }

    #[test]
    fn test_clone() {
        let mut m = HashMap::new();
        assert_eq!(m.len(), 0);
        assert!(m.insert(1, 2).is_none());
        assert_eq!(m.len(), 1);
        assert!(m.insert(2, 4).is_none());
        assert_eq!(m.len(), 2);
        let m2 = m.clone();
        assert_eq!(*m2.get(&1).unwrap(), 2);
        assert_eq!(*m2.get(&2).unwrap(), 4);
        assert_eq!(m2.len(), 2);
    }

    #[test]
    fn test_clone_from() {
        let mut m = HashMap::new();
        let mut m2 = HashMap::new();
        assert_eq!(m.len(), 0);
        assert!(m.insert(1, 2).is_none());
        assert_eq!(m.len(), 1);
        assert!(m.insert(2, 4).is_none());
        assert_eq!(m.len(), 2);
        m2.clone_from(&m);
        assert_eq!(*m2.get(&1).unwrap(), 2);
        assert_eq!(*m2.get(&2).unwrap(), 4);
        assert_eq!(m2.len(), 2);
    }

    thread_local! { static DROP_VECTOR: RefCell<Vec<i32>> = RefCell::new(Vec::new()) }

    #[derive(Hash, PartialEq, Eq)]
    struct Droppable {
        k: usize,
    }

    impl Droppable {
        fn new(k: usize) -> Droppable {
            DROP_VECTOR.with(|slot| {
                slot.borrow_mut()[k] += 1;
            });

            Droppable { k }
        }
    }

    impl Drop for Droppable {
        fn drop(&mut self) {
            DROP_VECTOR.with(|slot| {
                slot.borrow_mut()[self.k] -= 1;
            });
        }
    }

    impl Clone for Droppable {
        fn clone(&self) -> Self {
            Droppable::new(self.k)
        }
    }

    #[test]
    fn test_drops() {
        DROP_VECTOR.with(|slot| {
            *slot.borrow_mut() = vec![0; 200];
        });

        {
            let mut m = HashMap::new();

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 0);
                }
            });

            for i in 0..100 {
                let d1 = Droppable::new(i);
                let d2 = Droppable::new(i + 100);
                m.insert(d1, d2);
            }

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 1);
                }
            });

            for i in 0..50 {
                let k = Droppable::new(i);
                let v = m.remove(&k);

                assert!(v.is_some());

                DROP_VECTOR.with(|v| {
                    assert_eq!(v.borrow()[i], 1);
                    assert_eq!(v.borrow()[i + 100], 1);
                });
            }

            DROP_VECTOR.with(|v| {
                for i in 0..50 {
                    assert_eq!(v.borrow()[i], 0);
                    assert_eq!(v.borrow()[i + 100], 0);
                }

                for i in 50..100 {
                    assert_eq!(v.borrow()[i], 1);
                    assert_eq!(v.borrow()[i + 100], 1);
                }
            });
        }

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 0);
            }
        });
    }

    #[test]
    fn test_into_iter_drops() {
        DROP_VECTOR.with(|v| {
            *v.borrow_mut() = vec![0; 200];
        });

        let hm = {
            let mut hm = HashMap::new();

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 0);
                }
            });

            for i in 0..100 {
                let d1 = Droppable::new(i);
                let d2 = Droppable::new(i + 100);
                hm.insert(d1, d2);
            }

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 1);
                }
            });

            hm
        };

        // By the way, ensure that cloning doesn't screw up the dropping.
        drop(hm.clone());

        {
            let mut half = hm.into_iter().take(50);

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 1);
                }
            });

            for _ in half.by_ref() {}

            DROP_VECTOR.with(|v| {
                let nk = (0..100).filter(|&i| v.borrow()[i] == 1).count();

                let nv = (0..100).filter(|&i| v.borrow()[i + 100] == 1).count();

                assert_eq!(nk, 50);
                assert_eq!(nv, 50);
            });
        };

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 0);
            }
        });
    }

    #[test]
    fn test_empty_remove() {
        let mut m: HashMap<i32, bool> = HashMap::new();
        assert_eq!(m.remove(&0), None);
    }

    #[test]
    fn test_empty_entry() {
        let mut m: HashMap<i32, bool> = HashMap::new();
        match m.entry(0) {
            Occupied(_) => panic!(),
            Vacant(_) => {}
        }
        assert!(*m.entry(0).or_insert(true));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_empty_iter() {
        let mut m: HashMap<i32, bool> = HashMap::new();
        assert_eq!(m.drain().next(), None);
        assert_eq!(m.keys().next(), None);
        assert_eq!(m.values().next(), None);
        assert_eq!(m.values_mut().next(), None);
        assert_eq!(m.iter().next(), None);
        assert_eq!(m.iter_mut().next(), None);
        assert_eq!(m.len(), 0);
        assert!(m.is_empty());
        assert_eq!(m.into_iter().next(), None);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // FIXME: takes too long
    fn test_lots_of_insertions() {
        let mut m = HashMap::new();

        // Try this a few times to make sure we never screw up the hashmap's
        // internal state.
        for _ in 0..10 {
            assert!(m.is_empty());

            for i in 1..1001 {
                assert!(m.insert(i, i).is_none());

                for j in 1..=i {
                    let r = m.get(&j);
                    assert_eq!(r, Some(&j));
                }

                for j in i + 1..1001 {
                    let r = m.get(&j);
                    assert_eq!(r, None);
                }
            }

            for i in 1001..2001 {
                assert!(!m.contains_key(&i));
            }

            // remove forwards
            for i in 1..1001 {
                assert!(m.remove(&i).is_some());

                for j in 1..=i {
                    assert!(!m.contains_key(&j));
                }

                for j in i + 1..1001 {
                    assert!(m.contains_key(&j));
                }
            }

            for i in 1..1001 {
                assert!(!m.contains_key(&i));
            }

            for i in 1..1001 {
                assert!(m.insert(i, i).is_none());
            }

            // remove backwards
            for i in (1..1001).rev() {
                assert!(m.remove(&i).is_some());

                for j in i..1001 {
                    assert!(!m.contains_key(&j));
                }

                for j in 1..i {
                    assert!(m.contains_key(&j));
                }
            }
        }
    }

    #[test]
    fn test_find_mut() {
        let mut m = HashMap::new();
        assert!(m.insert(1, 12).is_none());
        assert!(m.insert(2, 8).is_none());
        assert!(m.insert(5, 14).is_none());
        let new = 100;
        match m.get_mut(&5) {
            None => panic!(),
            Some(x) => *x = new,
        }
        assert_eq!(m.get(&5), Some(&new));
    }

    #[test]
    fn test_insert_overwrite() {
        let mut m = HashMap::new();
        assert!(m.insert(1, 2).is_none());
        assert_eq!(*m.get(&1).unwrap(), 2);
        assert!(!m.insert(1, 3).is_none());
        assert_eq!(*m.get(&1).unwrap(), 3);
    }

    #[test]
    fn test_insert_conflicts() {
        let mut m = HashMap::with_capacity(4);
        assert!(m.insert(1, 2).is_none());
        assert!(m.insert(5, 3).is_none());
        assert!(m.insert(9, 4).is_none());
        assert_eq!(*m.get(&9).unwrap(), 4);
        assert_eq!(*m.get(&5).unwrap(), 3);
        assert_eq!(*m.get(&1).unwrap(), 2);
    }

    #[test]
    fn test_conflict_remove() {
        let mut m = HashMap::with_capacity(4);
        assert!(m.insert(1, 2).is_none());
        assert_eq!(*m.get(&1).unwrap(), 2);
        assert!(m.insert(5, 3).is_none());
        assert_eq!(*m.get(&1).unwrap(), 2);
        assert_eq!(*m.get(&5).unwrap(), 3);
        assert!(m.insert(9, 4).is_none());
        assert_eq!(*m.get(&1).unwrap(), 2);
        assert_eq!(*m.get(&5).unwrap(), 3);
        assert_eq!(*m.get(&9).unwrap(), 4);
        assert!(m.remove(&1).is_some());
        assert_eq!(*m.get(&9).unwrap(), 4);
        assert_eq!(*m.get(&5).unwrap(), 3);
    }

    #[test]
    fn test_is_empty() {
        let mut m = HashMap::with_capacity(4);
        assert!(m.insert(1, 2).is_none());
        assert!(!m.is_empty());
        assert!(m.remove(&1).is_some());
        assert!(m.is_empty());
    }

    #[test]
    fn test_remove() {
        let mut m = HashMap::new();
        m.insert(1, 2);
        assert_eq!(m.remove(&1), Some(2));
        assert_eq!(m.remove(&1), None);
    }

    #[test]
    fn test_remove_entry() {
        let mut m = HashMap::new();
        m.insert(1, 2);
        assert_eq!(m.remove_entry(&1), Some((1, 2)));
        assert_eq!(m.remove(&1), None);
    }

    #[test]
    fn test_iterate() {
        let mut m = HashMap::with_capacity(4);
        for i in 0..32 {
            assert!(m.insert(i, i * 2).is_none());
        }
        assert_eq!(m.len(), 32);

        let mut observed: u32 = 0;

        for (k, v) in &m {
            assert_eq!(*v, *k * 2);
            observed |= 1 << *k;
        }
        assert_eq!(observed, 0xFFFF_FFFF);
    }

    #[test]
    fn test_keys() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map: HashMap<_, _> = vec.into_iter().collect();
        let keys: Vec<_> = map.keys().cloned().collect();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn test_values() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map: HashMap<_, _> = vec.into_iter().collect();
        let values: Vec<_> = map.values().cloned().collect();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&'a'));
        assert!(values.contains(&'b'));
        assert!(values.contains(&'c'));
    }

    #[test]
    fn test_values_mut() {
        let vec = vec![(1, 1), (2, 2), (3, 3)];
        let mut map: HashMap<_, _> = vec.into_iter().collect();
        for value in map.values_mut() {
            *value = (*value) * 2
        }
        let values: Vec<_> = map.values().cloned().collect();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&2));
        assert!(values.contains(&4));
        assert!(values.contains(&6));
    }

    #[test]
    fn test_find() {
        let mut m = HashMap::new();
        assert!(m.get(&1).is_none());
        m.insert(1, 2);
        match m.get(&1) {
            None => panic!(),
            Some(v) => assert_eq!(*v, 2),
        }
    }

    #[test]
    fn test_eq() {
        let mut m1 = HashMap::new();
        m1.insert(1, 2);
        m1.insert(2, 3);
        m1.insert(3, 4);

        let mut m2 = HashMap::new();
        m2.insert(1, 2);
        m2.insert(2, 3);

        assert!(m1 != m2);

        m2.insert(3, 4);

        assert_eq!(m1, m2);
    }

    #[test]
    fn test_show() {
        let mut map = HashMap::new();
        let empty: HashMap<i32, i32> = HashMap::new();

        map.insert(1, 2);
        map.insert(3, 4);

        let map_str = format!("{:?}", map);

        assert!(map_str == "{1: 2, 3: 4}" || map_str == "{3: 4, 1: 2}");
        assert_eq!(format!("{:?}", empty), "{}");
    }

    #[test]
    fn test_expand() {
        let mut m = HashMap::new();

        assert_eq!(m.len(), 0);
        assert!(m.is_empty());

        let mut i = 0;
        let old_raw_cap = m.raw_capacity();
        while old_raw_cap == m.raw_capacity() {
            m.insert(i, i);
            i += 1;
        }

        assert_eq!(m.len(), i);
        assert!(!m.is_empty());
    }

    #[test]
    fn test_behavior_resize_policy() {
        let mut m = HashMap::new();

        assert_eq!(m.len(), 0);
        assert_eq!(m.raw_capacity(), 1);
        assert!(m.is_empty());

        m.insert(0, 0);
        m.remove(&0);
        assert!(m.is_empty());
        let initial_raw_cap = m.raw_capacity();
        m.reserve(initial_raw_cap);
        let raw_cap = m.raw_capacity();

        assert_eq!(raw_cap, initial_raw_cap * 2);

        let mut i = 0;
        for _ in 0..raw_cap * 3 / 4 {
            m.insert(i, i);
            i += 1;
        }
        // three quarters full

        assert_eq!(m.len(), i);
        assert_eq!(m.raw_capacity(), raw_cap);

        for _ in 0..raw_cap / 4 {
            m.insert(i, i);
            i += 1;
        }
        // half full

        let new_raw_cap = m.raw_capacity();
        assert_eq!(new_raw_cap, raw_cap * 2);

        for _ in 0..raw_cap / 2 - 1 {
            i -= 1;
            m.remove(&i);
            assert_eq!(m.raw_capacity(), new_raw_cap);
        }
        // A little more than one quarter full.
        m.shrink_to_fit();
        assert_eq!(m.raw_capacity(), raw_cap);
        // again, a little more than half full
        for _ in 0..raw_cap / 2 {
            i -= 1;
            m.remove(&i);
        }
        m.shrink_to_fit();

        assert_eq!(m.len(), i);
        assert!(!m.is_empty());
        assert_eq!(m.raw_capacity(), initial_raw_cap);
    }

    #[test]
    fn test_reserve_shrink_to_fit() {
        let mut m = HashMap::new();
        m.insert(0, 0);
        m.remove(&0);
        assert!(m.capacity() >= m.len());
        for i in 0..128 {
            m.insert(i, i);
        }
        m.reserve(256);

        let usable_cap = m.capacity();
        for i in 128..(128 + 256) {
            m.insert(i, i);
            assert_eq!(m.capacity(), usable_cap);
        }

        for i in 100..(128 + 256) {
            assert_eq!(m.remove(&i), Some(i));
        }
        m.shrink_to_fit();

        assert_eq!(m.len(), 100);
        assert!(!m.is_empty());
        assert!(m.capacity() >= m.len());

        for i in 0..100 {
            assert_eq!(m.remove(&i), Some(i));
        }
        m.shrink_to_fit();
        m.insert(0, 0);

        assert_eq!(m.len(), 1);
        assert!(m.capacity() >= m.len());
        assert_eq!(m.remove(&0), Some(0));
    }

    #[test]
    fn test_from_iter() {
        let xs = [(1, 1), (2, 2), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: HashMap<_, _> = xs.iter().cloned().collect();

        for &(k, v) in &xs {
            assert_eq!(map.get(&k), Some(&v));
        }

        assert_eq!(map.iter().len(), xs.len() - 1);
    }

    #[test]
    fn test_size_hint() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: HashMap<_, _> = xs.iter().cloned().collect();

        let mut iter = map.iter();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_iter_len() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: HashMap<_, _> = xs.iter().cloned().collect();

        let mut iter = map.iter();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn test_mut_size_hint() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let mut map: HashMap<_, _> = xs.iter().cloned().collect();

        let mut iter = map.iter_mut();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_iter_mut_len() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let mut map: HashMap<_, _> = xs.iter().cloned().collect();

        let mut iter = map.iter_mut();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn test_index() {
        let mut map = HashMap::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        assert_eq!(map[&2], 1);
    }

    #[test]
    #[should_panic]
    fn test_index_nonexistent() {
        let mut map = HashMap::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        map[&4];
    }

    #[test]
    fn test_entry() {
        let xs = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

        let mut map: HashMap<_, _> = xs.iter().cloned().collect();

        // Existing key (insert)
        match map.entry(1) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                assert_eq!(view.get(), &10);
                assert_eq!(view.insert(100), 10);
            }
        }
        assert_eq!(map.get(&1).unwrap(), &100);
        assert_eq!(map.len(), 6);

        // Existing key (update)
        match map.entry(2) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                let v = view.get_mut();
                let new_v = (*v) * 10;
                *v = new_v;
            }
        }
        assert_eq!(map.get(&2).unwrap(), &200);
        assert_eq!(map.len(), 6);

        // Existing key (take)
        match map.entry(3) {
            Vacant(_) => unreachable!(),
            Occupied(view) => {
                assert_eq!(view.remove(), 30);
            }
        }
        assert_eq!(map.get(&3), None);
        assert_eq!(map.len(), 5);

        // Inexistent key (insert)
        match map.entry(10) {
            Occupied(_) => unreachable!(),
            Vacant(view) => {
                assert_eq!(*view.insert(1000), 1000);
            }
        }
        assert_eq!(map.get(&10).unwrap(), &1000);
        assert_eq!(map.len(), 6);
    }

    #[test]
    fn test_entry_take_doesnt_corrupt() {
        #![allow(deprecated)] //rand
                              // Test for #19292
        fn check(m: &HashMap<i32, ()>) {
            for k in m.keys() {
                assert!(m.contains_key(k), "{} is in keys() but not in the map?", k);
            }
        }

        let mut m = HashMap::new();

        let mut rng = {
            let seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            SmallRng::from_seed(seed)
        };

        // Populate the map with some items.
        for _ in 0..50 {
            let x = rng.gen_range(-10, 10);
            m.insert(x, ());
        }

        for _ in 0..1000 {
            let x = rng.gen_range(-10, 10);
            match m.entry(x) {
                Vacant(_) => {}
                Occupied(e) => {
                    e.remove();
                }
            }

            check(&m);
        }
    }

    #[test]
    fn test_extend_ref() {
        let mut a = HashMap::new();
        a.insert(1, "one");
        let mut b = HashMap::new();
        b.insert(2, "two");
        b.insert(3, "three");

        a.extend(&b);

        assert_eq!(a.len(), 3);
        assert_eq!(a[&1], "one");
        assert_eq!(a[&2], "two");
        assert_eq!(a[&3], "three");
    }

    #[test]
    fn test_capacity_not_less_than_len() {
        let mut a = HashMap::new();
        let mut item = 0;

        for _ in 0..116 {
            a.insert(item, 0);
            item += 1;
        }

        assert!(a.capacity() > a.len());

        let free = a.capacity() - a.len();
        for _ in 0..free {
            a.insert(item, 0);
            item += 1;
        }

        assert_eq!(a.len(), a.capacity());

        // Insert at capacity should cause allocation.
        a.insert(item, 0);
        assert!(a.capacity() > a.len());
    }

    #[test]
    fn test_occupied_entry_key() {
        let mut a = HashMap::new();
        let key = "hello there";
        let value = "value goes here";
        assert!(a.is_empty());
        a.insert(key.clone(), value.clone());
        assert_eq!(a.len(), 1);
        assert_eq!(a[key], value);

        match a.entry(key.clone()) {
            Vacant(_) => panic!(),
            Occupied(e) => assert_eq!(key, *e.key()),
        }
        assert_eq!(a.len(), 1);
        assert_eq!(a[key], value);
    }

    #[test]
    fn test_vacant_entry_key() {
        let mut a = HashMap::new();
        let key = "hello there";
        let value = "value goes here";

        assert!(a.is_empty());
        match a.entry(key.clone()) {
            Occupied(_) => panic!(),
            Vacant(e) => {
                assert_eq!(key, *e.key());
                e.insert(value.clone());
            }
        }
        assert_eq!(a.len(), 1);
        assert_eq!(a[key], value);
    }

    #[test]
    fn test_occupied_entry_replace_entry_with() {
        let mut a = HashMap::new();

        let key = "a key";
        let value = "an initial value";
        let new_value = "a new value";

        let entry = a.entry(key).insert(value).replace_entry_with(|k, v| {
            assert_eq!(k, &key);
            assert_eq!(v, value);
            Some(new_value)
        });

        match entry {
            Occupied(e) => {
                assert_eq!(e.key(), &key);
                assert_eq!(e.get(), &new_value);
            }
            Vacant(_) => panic!(),
        }

        assert_eq!(a[key], new_value);
        assert_eq!(a.len(), 1);

        let entry = match a.entry(key) {
            Occupied(e) => e.replace_entry_with(|k, v| {
                assert_eq!(k, &key);
                assert_eq!(v, new_value);
                None
            }),
            Vacant(_) => panic!(),
        };

        match entry {
            Vacant(e) => assert_eq!(e.key(), &key),
            Occupied(_) => panic!(),
        }

        assert!(!a.contains_key(key));
        assert_eq!(a.len(), 0);
    }

    #[test]
    fn test_entry_and_replace_entry_with() {
        let mut a = HashMap::new();

        let key = "a key";
        let value = "an initial value";
        let new_value = "a new value";

        let entry = a.entry(key).and_replace_entry_with(|_, _| panic!());

        match entry {
            Vacant(e) => assert_eq!(e.key(), &key),
            Occupied(_) => panic!(),
        }

        a.insert(key, value);

        let entry = a.entry(key).and_replace_entry_with(|k, v| {
            assert_eq!(k, &key);
            assert_eq!(v, value);
            Some(new_value)
        });

        match entry {
            Occupied(e) => {
                assert_eq!(e.key(), &key);
                assert_eq!(e.get(), &new_value);
            }
            Vacant(_) => panic!(),
        }

        assert_eq!(a[key], new_value);
        assert_eq!(a.len(), 1);

        let entry = a.entry(key).and_replace_entry_with(|k, v| {
            assert_eq!(k, &key);
            assert_eq!(v, new_value);
            None
        });

        match entry {
            Vacant(e) => assert_eq!(e.key(), &key),
            Occupied(_) => panic!(),
        }

        assert!(!a.contains_key(key));
        assert_eq!(a.len(), 0);
    }

    #[test]
    fn test_raw_occupied_entry_replace_entry_with() {
        let mut a = HashMap::new();

        let key = "a key";
        let value = "an initial value";
        let new_value = "a new value";

        let entry = a
            .raw_entry_mut()
            .from_key(&key)
            .insert(key, value)
            .replace_entry_with(|k, v| {
                assert_eq!(k, &key);
                assert_eq!(v, value);
                Some(new_value)
            });

        match entry {
            RawEntryMut::Occupied(e) => {
                assert_eq!(e.key(), &key);
                assert_eq!(e.get(), &new_value);
            }
            RawEntryMut::Vacant(_) => panic!(),
        }

        assert_eq!(a[key], new_value);
        assert_eq!(a.len(), 1);

        let entry = match a.raw_entry_mut().from_key(&key) {
            RawEntryMut::Occupied(e) => e.replace_entry_with(|k, v| {
                assert_eq!(k, &key);
                assert_eq!(v, new_value);
                None
            }),
            RawEntryMut::Vacant(_) => panic!(),
        };

        match entry {
            RawEntryMut::Vacant(_) => {}
            RawEntryMut::Occupied(_) => panic!(),
        }

        assert!(!a.contains_key(key));
        assert_eq!(a.len(), 0);
    }

    #[test]
    fn test_raw_entry_and_replace_entry_with() {
        let mut a = HashMap::new();

        let key = "a key";
        let value = "an initial value";
        let new_value = "a new value";

        let entry = a
            .raw_entry_mut()
            .from_key(&key)
            .and_replace_entry_with(|_, _| panic!());

        match entry {
            RawEntryMut::Vacant(_) => {}
            RawEntryMut::Occupied(_) => panic!(),
        }

        a.insert(key, value);

        let entry = a
            .raw_entry_mut()
            .from_key(&key)
            .and_replace_entry_with(|k, v| {
                assert_eq!(k, &key);
                assert_eq!(v, value);
                Some(new_value)
            });

        match entry {
            RawEntryMut::Occupied(e) => {
                assert_eq!(e.key(), &key);
                assert_eq!(e.get(), &new_value);
            }
            RawEntryMut::Vacant(_) => panic!(),
        }

        assert_eq!(a[key], new_value);
        assert_eq!(a.len(), 1);

        let entry = a
            .raw_entry_mut()
            .from_key(&key)
            .and_replace_entry_with(|k, v| {
                assert_eq!(k, &key);
                assert_eq!(v, new_value);
                None
            });

        match entry {
            RawEntryMut::Vacant(_) => {}
            RawEntryMut::Occupied(_) => panic!(),
        }

        assert!(!a.contains_key(key));
        assert_eq!(a.len(), 0);
    }

    #[test]
    fn test_replace_entry_with_doesnt_corrupt() {
        #![allow(deprecated)] //rand
                              // Test for #19292
        fn check(m: &HashMap<i32, ()>) {
            for k in m.keys() {
                assert!(m.contains_key(k), "{} is in keys() but not in the map?", k);
            }
        }

        let mut m = HashMap::new();

        let mut rng = {
            let seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            SmallRng::from_seed(seed)
        };

        // Populate the map with some items.
        for _ in 0..50 {
            let x = rng.gen_range(-10, 10);
            m.insert(x, ());
        }

        for _ in 0..1000 {
            let x = rng.gen_range(-10, 10);
            m.entry(x).and_replace_entry_with(|_, _| None);
            check(&m);
        }
    }

    #[test]
    fn test_retain() {
        let mut map: HashMap<i32, i32> = (0..100).map(|x| (x, x * 10)).collect();

        map.retain(|&k, _| k % 2 == 0);
        assert_eq!(map.len(), 50);
        assert_eq!(map[&2], 20);
        assert_eq!(map[&4], 40);
        assert_eq!(map[&6], 60);
    }

    #[test]
    fn test_drain_filter() {
        {
            let mut map: HashMap<i32, i32> = (0..8).map(|x| (x, x * 10)).collect();
            let drained = map.drain_filter(|&k, _| k % 2 == 0);
            let mut out = drained.collect::<Vec<_>>();
            out.sort_unstable();
            assert_eq!(vec![(0, 0), (2, 20), (4, 40), (6, 60)], out);
            assert_eq!(map.len(), 4);
        }
        {
            let mut map: HashMap<i32, i32> = (0..8).map(|x| (x, x * 10)).collect();
            drop(map.drain_filter(|&k, _| k % 2 == 0));
            assert_eq!(map.len(), 4);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // FIXME: no OOM signalling (https://github.com/rust-lang/miri/issues/613)
    fn test_try_reserve() {
        let mut empty_bytes: HashMap<u8, u8> = HashMap::new();

        const MAX_USIZE: usize = usize::MAX;

        if let Err(CapacityOverflow) = empty_bytes.try_reserve(MAX_USIZE) {
        } else {
            panic!("usize::MAX should trigger an overflow!");
        }

        if let Err(AllocError { .. }) = empty_bytes.try_reserve(MAX_USIZE / 8) {
        } else {
            // This may succeed if there is enough free memory. Attempt to
            // allocate a second hashmap to ensure the allocation will fail.
            let mut empty_bytes2: HashMap<u8, u8> = HashMap::new();
            if let Err(AllocError { .. }) = empty_bytes2.try_reserve(MAX_USIZE / 8) {
            } else {
                panic!("usize::MAX / 8 should trigger an OOM!");
            }
        }
    }

    #[test]
    fn test_raw_entry() {
        use super::RawEntryMut::{Occupied, Vacant};

        let xs = [(1i32, 10i32), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

        let mut map: HashMap<_, _> = xs.iter().cloned().collect();

        let compute_hash = |map: &HashMap<i32, i32>, k: i32| -> u64 {
            super::make_insert_hash::<i32, _>(map.hasher(), &k)
        };

        // Existing key (insert)
        match map.raw_entry_mut().from_key(&1) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                assert_eq!(view.get(), &10);
                assert_eq!(view.insert(100), 10);
            }
        }
        let hash1 = compute_hash(&map, 1);
        assert_eq!(map.raw_entry().from_key(&1).unwrap(), (&1, &100));
        assert_eq!(
            map.raw_entry().from_hash(hash1, |k| *k == 1).unwrap(),
            (&1, &100)
        );
        assert_eq!(
            map.raw_entry().from_key_hashed_nocheck(hash1, &1).unwrap(),
            (&1, &100)
        );
        assert_eq!(map.len(), 6);

        // Existing key (update)
        match map.raw_entry_mut().from_key(&2) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                let v = view.get_mut();
                let new_v = (*v) * 10;
                *v = new_v;
            }
        }
        let hash2 = compute_hash(&map, 2);
        assert_eq!(map.raw_entry().from_key(&2).unwrap(), (&2, &200));
        assert_eq!(
            map.raw_entry().from_hash(hash2, |k| *k == 2).unwrap(),
            (&2, &200)
        );
        assert_eq!(
            map.raw_entry().from_key_hashed_nocheck(hash2, &2).unwrap(),
            (&2, &200)
        );
        assert_eq!(map.len(), 6);

        // Existing key (take)
        let hash3 = compute_hash(&map, 3);
        match map.raw_entry_mut().from_key_hashed_nocheck(hash3, &3) {
            Vacant(_) => unreachable!(),
            Occupied(view) => {
                assert_eq!(view.remove_entry(), (3, 30));
            }
        }
        assert_eq!(map.raw_entry().from_key(&3), None);
        assert_eq!(map.raw_entry().from_hash(hash3, |k| *k == 3), None);
        assert_eq!(map.raw_entry().from_key_hashed_nocheck(hash3, &3), None);
        assert_eq!(map.len(), 5);

        // Nonexistent key (insert)
        match map.raw_entry_mut().from_key(&10) {
            Occupied(_) => unreachable!(),
            Vacant(view) => {
                assert_eq!(view.insert(10, 1000), (&mut 10, &mut 1000));
            }
        }
        assert_eq!(map.raw_entry().from_key(&10).unwrap(), (&10, &1000));
        assert_eq!(map.len(), 6);

        // Ensure all lookup methods produce equivalent results.
        for k in 0..12 {
            let hash = compute_hash(&map, k);
            let v = map.get(&k).cloned();
            let kv = v.as_ref().map(|v| (&k, v));

            assert_eq!(map.raw_entry().from_key(&k), kv);
            assert_eq!(map.raw_entry().from_hash(hash, |q| *q == k), kv);
            assert_eq!(map.raw_entry().from_key_hashed_nocheck(hash, &k), kv);

            match map.raw_entry_mut().from_key(&k) {
                Occupied(mut o) => assert_eq!(Some(o.get_key_value()), kv),
                Vacant(_) => assert_eq!(v, None),
            }
            match map.raw_entry_mut().from_key_hashed_nocheck(hash, &k) {
                Occupied(mut o) => assert_eq!(Some(o.get_key_value()), kv),
                Vacant(_) => assert_eq!(v, None),
            }
            match map.raw_entry_mut().from_hash(hash, |q| *q == k) {
                Occupied(mut o) => assert_eq!(Some(o.get_key_value()), kv),
                Vacant(_) => assert_eq!(v, None),
            }
        }
    }

    #[test]
    fn test_key_without_hash_impl() {
        #[derive(Debug)]
        struct IntWrapper(u64);

        let mut m: HashMap<IntWrapper, (), ()> = HashMap::default();
        {
            assert!(m.raw_entry().from_hash(0, |k| k.0 == 0).is_none());
        }
        {
            let vacant_entry = match m.raw_entry_mut().from_hash(0, |k| k.0 == 0) {
                RawEntryMut::Occupied(..) => panic!("Found entry for key 0"),
                RawEntryMut::Vacant(e) => e,
            };
            vacant_entry.insert_with_hasher(0, IntWrapper(0), (), |k| k.0);
        }
        {
            assert!(m.raw_entry().from_hash(0, |k| k.0 == 0).is_some());
            assert!(m.raw_entry().from_hash(1, |k| k.0 == 1).is_none());
            assert!(m.raw_entry().from_hash(2, |k| k.0 == 2).is_none());
        }
        {
            let vacant_entry = match m.raw_entry_mut().from_hash(1, |k| k.0 == 1) {
                RawEntryMut::Occupied(..) => panic!("Found entry for key 1"),
                RawEntryMut::Vacant(e) => e,
            };
            vacant_entry.insert_with_hasher(1, IntWrapper(1), (), |k| k.0);
        }
        {
            assert!(m.raw_entry().from_hash(0, |k| k.0 == 0).is_some());
            assert!(m.raw_entry().from_hash(1, |k| k.0 == 1).is_some());
            assert!(m.raw_entry().from_hash(2, |k| k.0 == 2).is_none());
        }
        {
            let occupied_entry = match m.raw_entry_mut().from_hash(0, |k| k.0 == 0) {
                RawEntryMut::Occupied(e) => e,
                RawEntryMut::Vacant(..) => panic!("Couldn't find entry for key 0"),
            };
            occupied_entry.remove();
        }
        assert!(m.raw_entry().from_hash(0, |k| k.0 == 0).is_none());
        assert!(m.raw_entry().from_hash(1, |k| k.0 == 1).is_some());
        assert!(m.raw_entry().from_hash(2, |k| k.0 == 2).is_none());
    }

    #[test]
    #[cfg(feature = "raw")]
    fn test_into_iter_refresh() {
        use core::hash::{BuildHasher, Hash, Hasher};

        #[cfg(miri)]
        const N: usize = 32;
        #[cfg(not(miri))]
        const N: usize = 128;

        let mut rng = rand::thread_rng();
        for n in 0..N {
            let mut m = HashMap::new();
            for i in 0..n {
                assert!(m.insert(i, 2 * i).is_none());
            }
            let hasher = m.hasher().clone();

            let mut it = unsafe { m.table.iter() };
            assert_eq!(it.len(), n);

            let mut i = 0;
            let mut left = n;
            let mut removed = Vec::new();
            loop {
                // occasionally remove some elements
                if i < n && rng.gen_bool(0.1) {
                    let mut hsh = hasher.build_hasher();
                    i.hash(&mut hsh);
                    let hash = hsh.finish();

                    unsafe {
                        let e = m.table.find(hash, |q| q.0.eq(&i));
                        if let Some(e) = e {
                            it.reflect_remove(&e);
                            let t = m.table.remove(e);
                            removed.push(t);
                            left -= 1;
                        } else {
                            assert!(removed.contains(&(i, 2 * i)), "{} not in {:?}", i, removed);
                            let e = m.table.insert(
                                hash,
                                (i, 2 * i),
                                super::make_hasher::<usize, _, usize, _>(&hasher),
                            );
                            it.reflect_insert(&e);
                            if let Some(p) = removed.iter().position(|e| e == &(i, 2 * i)) {
                                removed.swap_remove(p);
                            }
                            left += 1;
                        }
                    }
                }

                let e = it.next();
                if e.is_none() {
                    break;
                }
                assert!(i < n);
                let t = unsafe { e.unwrap().as_ref() };
                assert!(!removed.contains(t));
                let (k, v) = t;
                assert_eq!(*v, 2 * k);
                i += 1;
            }
            assert!(i <= n);

            // just for safety:
            assert_eq!(m.table.len(), left);
        }
    }

    #[test]
    fn test_const_with_hasher() {
        use core::hash::BuildHasher;
        use std::collections::hash_map::DefaultHasher;

        #[derive(Clone)]
        struct MyHasher;
        impl BuildHasher for MyHasher {
            type Hasher = DefaultHasher;

            fn build_hasher(&self) -> DefaultHasher {
                DefaultHasher::new()
            }
        }

        const EMPTY_MAP: HashMap<u32, std::string::String, MyHasher> =
            HashMap::with_hasher(MyHasher);

        let mut map = EMPTY_MAP.clone();
        map.insert(17, "seventeen".to_owned());
        assert_eq!("seventeen", map[&17]);
    }

    #[test]
    #[cfg(feature = "nightly")]
    fn test_get_each_mut() {
        use crate::UnavailableMutError::*;

        let mut map = HashMap::new();
        map.insert("foo".to_owned(), 0);
        map.insert("bar".to_owned(), 10);
        map.insert("baz".to_owned(), 20);
        map.insert("qux".to_owned(), 30);

        let xs = map.get_each_mut(["foo", "dud", "foo", "qux"]);
        assert_eq!(
            xs,
            [Ok(&mut 0), Err(Absent), Err(Duplicate(0)), Ok(&mut 30)]
        );

        let ys = map.get_each_key_value_mut(["bar", "baz", "baz", "dip"]);
        assert_eq!(
            ys,
            [
                Ok((&"bar".to_owned(), &mut 10)),
                Ok((&"baz".to_owned(), &mut 20)),
                Err(Duplicate(1)),
                Err(Absent),
            ]
        );
    }
}
