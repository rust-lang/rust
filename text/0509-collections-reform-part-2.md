- Start Date: 2014-12-18
- RFC PR: https://github.com/rust-lang/rfcs/pull/509
- Rust Issue: https://github.com/rust-lang/rust/issues/19986

# Summary

This RFC shores up the finer details of collections reform. In particular, where the
[previous RFC][part1]
focused on general conventions and patterns, this RFC focuses on specific APIs. It also patches
up any errors that were found during implementation of [part 1][part1]. Some of these changes
have already been implemented, and simply need to be ratified.

# Motivation

Collections reform stabilizes "standard" interfaces, but there's a lot that still needs to be
hashed out.

# Detailed design

## The fate of entire collections:

* Stable: Vec, RingBuf, HashMap, HashSet, BTreeMap, BTreeSet, DList, BinaryHeap
* Unstable: Bitv, BitvSet, VecMap
* Move to [collect-rs](https://github.com/Gankro/collect-rs/) for incubation:
EnumSet, bitflags!, LruCache, TreeMap, TreeSet, TrieMap, TrieSet

The stable collections have solid implementations, well-maintained APIs, are non-trivial,
fundamental, and clearly useful.

The unstable collections are effectively "on probation". They're ok, but they need some TLC and
further consideration before we commit to having them in the standard library *forever*. Bitv in
particular won't have *quite* the right API without IndexGet *and* IndexSet.

The collections being moved out are in poor shape. EnumSet is weird/trivial, bitflags is awkward,
LruCache is niche. Meanwhile Tree\* and Trie\* have simply bit-rotted for too long, without anyone
clearly stepping up to maintain them. Their code is scary, and their APIs are out of date. Their
functionality can also already reasonably be obtained through either HashMap or BTreeMap.

Of course, instead of moving them out-of-tree, they could be left `experimental`, but that would
perhaps be a fate *worse* than death, as it would mean that these collections would only be
accessible to those who opt into running the Rust nightly. This way, these collections will be
available for everyone through the cargo ecosystem. Putting them in `collect-rs` also gives them
a chance to still benefit from a network effect and active experimentation. If they thrive there,
they may still return to the standard library at a later time.

## Add the following methods:

* To all collections
```
/// Moves all the elements of `other` into `Self`, leaving `other` empty.
pub fn append(&mut self, other: &mut Self)
```

Collections know everything about themselves, and can therefore move data more
efficiently than any more generic mechanism. Vec's can safely trust their own capacity
and length claims. DList and TreeMap can also reuse nodes, avoiding allocating.

This is by-ref instead of by-value for a couple reasons. First, it adds symmetry (one doesn't have
to be owned). Second, in the case of array-based structures, it allows `other`'s capacity to be
reused. This shouldn't have much expense in the way of making `other` valid, as almost all of our
collections are basically a no-op to make an empty version of if necessary (usually it amounts to
zeroing a few words of memory). BTree is the only exception the author is aware of (root is pre-
allocated
to avoid an Option).

* To DList, Vec, RingBuf, BitV:
```
/// Splits the collection into two at the given index. Useful for similar reasons as `append`.
pub fn split_off(&mut self, at: uint) -> Self;
```

* To all other "sorted" collections
```
/// Splits the collection into two at the given key. Returns everything after the given key,
/// including the key.
pub fn split_off<B: Borrow<K>>(&mut self, at: B) -> Self;
```

Similar reasoning to `append`, although perhaps even more needed, as there's *no* other mechanism
for moving an entire subrange of a collection efficiently like this. `into_iterator` consumes
the whole collection, and using `remove` methods will do a lot of unnecessary work. For instance,
in the case of `Vec`, using `pop` and `push` will involve many length changes, bounds checks,
unwraps, and ultimately produce a *reversed* Vec.

* To BitvSet, VecMap:

```
/// Reserves capacity for an element to be inserted at `len - 1` in the given
/// collection. The collection may reserve more space to avoid frequent reallocations.
pub fn reserve_len(&mut self, len: uint)

/// Reserves the minimum capacity for an element to be inserted at `len - 1` in the given
/// collection.
pub fn reserve_len_exact(&mut self, len: uint)
```

The "capacity" of these two collections isn't really strongly related to the
number of elements they hold, but rather the largest index an element is stored at.
See Errata and Alternatives for extended discussion of this design.

* For Ringbuf:
```
/// Gets two slices that cover the whole range of the RingBuf.
/// The second one may be empty. Otherwise, it continues *after* the first.
pub fn as_slices(&'a self) -> (&'a [T], &'a [T])
```

This provides some amount of support for viewing the RingBuf like a slice. Unfortunately
the RingBuf may be wrapped, making this impossible. See Alternatives for other designs.

There is an implementation of this at rust-lang/rust#19903.

* For Vec:
```
/// Resizes the `Vec` in-place so that `len()` equals to `new_len`.
///
/// Calls either `grow()` or `truncate()` depending on whether `new_len`
/// is larger than the current value of `len()` or not.
pub fn resize(&mut self, new_len: uint, value: T) where T: Clone
```

This is actually easy to implement out-of-tree on top of the current Vec API, but it has
been frequently requested.

* For Vec, RingBuf, BinaryHeap, HashMap and HashSet:
```
/// Clears the container, returning its owned contents as an iterator, but keeps the
/// allocated memory for reuse.
pub fn drain(&mut self) -> Drain<T>;
```

This provides a way to grab elements out of a collection by value, without
deallocating the storage for the collection itself.

There is a partial implementation of this at rust-lang/rust#19946.

==============
## Deprecate

* `Vec::from_fn(n, f)` use `(0..n).map(f).collect()`
* `Vec::from_elem(n, v)` use `repeat(v).take(n).collect()`
* `Vec::grow` use `extend(repeat(v).take(n))`
* `Vec::grow_fn` use `extend((0..n).map(f))`
* `dlist::ListInsertion` in favour of inherent methods on the iterator

==============

## Misc Stabilization:

* Rename `BinaryHeap::top` to `BinaryHeap::peek`. `peek` is a more clear name than `top`, and is
already used elsewhere in our APIs.

* `Bitv::get`, `Bitv::set`, where `set` panics on OOB, and `get` returns an Option. `set` may want
to wait on IndexSet being a thing (see Alternatives).

* Rename SmallIntMap to VecMap. (already done)

* Stabilize `front`/`back`/`front_mut`/`back_mut` for peeking on the ends of Deques

* Explicitly specify HashMap's iterators to be non-deterministic between iterations. This would
allow e.g. `next_back` to be implemented as `next`, reducing code complexity. This can be undone
in the future backwards-compatibly, but the reverse does not hold.

* Move `Vec` from `std::vec` to `std::collections::vec`.

* Stabilize RingBuf::swap

==============

## Clarifications and Errata from Part 1

* Not every collection can implement every kind of iterator. This RFC simply wishes to clarify
that iterator implementation should be a "best effort" for what makes sense for the collection.

* Bitv was marked as having *explicit* growth capacity semantics, when in fact it is implicit
growth. It has the same semantics as Vec.

* BitvSet and VecMap are part of a surprise *fourth* capacity class, which isn't really based on
the number of elements contained, but on the maximum index stored. This RFC proposes the name of
*maximum growth*.

* `reserve(x)` should specifically reserve space for `x + len()` elements, as opposed to e.g. `x +
capacity()` elements.

* Capacity methods should be based on a "best effort" model:

    * `capacity()` can be regarded as a *lower bound* on the number of elements that can be
    inserted before a resize occurs. It is acceptable for more elements to be insertable. A
    collection may also randomly resize before capacity is met if highly degenerate behaviour
    occurs. This is relevant to HashMap, which due to its use of integer multiplication cannot
    precisely compute its "true" capacity. It also may wish to resize early if a long chain of
    collisions occurs. Note that Vec should make *clear* guarantees about the precision of
    capacity, as this is important for `unsafe` usage.

    * `reserve_exact` may be subverted by the collection's own requirements (e.g. many collections
    require a capacity related to a power of two for fast modular arithmetic). The allocator may
    also give the collection more space than it requests, in which case it may as well use that
    space. It will still give you at least as much capacity as you request.

    * `shrink_to_fit` may not shrink to the true minimum size for similar reasons as
    `reserve_exact`.

    * Neither `reserve` nor `reserve_exact` can be trusted to reliably produce a specific
    capacity. At best you can guarantee that there will be space for the number you ask for.
    Although even then `capacity` itself may return a smaller number due to its own fuzziness.

==============

## Entry API V2.0

The old Entry API:
```
impl Map<K, V> {
    fn entry<'a>(&'a mut self, key: K) -> Entry<'a, K, V>
}

pub enum Entry<'a, K: 'a, V: 'a> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

impl<'a, K, V> VacantEntry<'a, K, V> {
    fn set(self, value: V) -> &'a mut V
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    fn get(&self) -> &V
    fn get_mut(&mut self) -> &mut V
    fn into_mut(self) -> &'a mut V
    fn set(&mut self, value: V) -> V
    fn take(self) -> V
}
```

Based on feedback and collections reform landing, this RFC proposes the following new API:

```
impl Map<K, V> {
    fn entry<'a, O: ToOwned<K>>(&'a mut self, key: &O) -> Entry<'a, O, V>
}

pub enum Entry<'a, O: 'a, V: 'a> {
    Occupied(OccupiedEntry<'a, O, V>),
    Vacant(VacantEntry<'a, O, V>),
}

impl Entry<'a, O: 'a, V:'a> {
    fn get(self) -> Result<&'a mut V, VacantEntry<'a, O, V>>
}

impl<'a, K, V> VacantEntry<'a, K, V> {
    fn insert(self, value: V) -> &'a mut V
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    fn get(&self) -> &V
    fn get_mut(&mut self) -> &mut V
    fn into_mut(self) -> &'a mut V
    fn insert(&mut self, value: V) -> V
    fn remove(self) -> V
}
```

Replacing get/get_mut with Deref is simply a nice ergonomic improvement. Renaming `set` and `take`
to `insert` and `remove` brings the API more inline with other collection APIs, and makes it
more clear what they do. The convenience method on Entry itself makes it just nicer to use.
Permitting the following `map.entry(key).get().or_else(|vacant| vacant.insert(Vec::new()))`.

This API should be stabilized for 1.0 with the exception of the impl on Entry itself.

# Alternatives

## Traits vs Inherent Impls on Entries
The Entry API as proposed would leave Entry and its two variants defined by each collection. We
could instead make the actual concrete VacantEntry/OccupiedEntry implementors implement a trait.
This would allow Entry to be hoisted up to root of collections, with utility functions implemented
once, as well as only requiring one import when using multiple collections. This *would* require
that the traits be imported, unless we get inherent trait implementations.

These traits can of course be introduced later.

==============

## Alternatives to ToOwned on Entries
The Entry API currently is a bit wasteful in the by-value key case. If, for instance, a user of a
`HashMap<String, _>` happens to have a String they don't mind losing, they can't pass the String by
-value to the Map. They must pass it by-reference, and have it get cloned.

One solution to this is to actually have the bound be IntoCow. This will potentially have some
runtime overhead, but it should be dwarfed by the cost of an insertion anyway, and would be a
clear win in the by-value case.

Another alternative would be an *IntoOwned* trait, which would have the signature `(self) ->
Owned`, as opposed to the current ToOwned `(&self) -> Owned`. IntoOwned more closely matches the
semantics we actually want for our entry keys, because we really don't care about preserving them
after the conversion. This would allow us to dispatch to either a no-op or a full clone as
necessary. This trait would also be appropriate for the CoW type, and in fact all of our current
uses of the type. However the relationship between FromBorrow and IntoOwned is currently awkward
to express with our type system, as it would have to be implemented e.g. for `&str` instead of
`str`. IntoOwned also has trouble co-existing "fully" with ToOwned due to current lack of negative
bounds in where clauses. That is, we would want a blanket impl of IntoOwned for ToOwned, but this
can't be properly expressed for coherence reasons.

This RFC does not propose either of these designs in favour of choosing the conservative ToOwned
now, with the possibility of "upgrading" into IntoOwned, IntoCow, or something else when we have a
better view of the type-system landscape.

==============

## Don't stabilize `Bitv::set`

We could wait for IndexSet, Or make `set` return a result.
`set` really is redundant with an IndexSet implementation, and we
don't like to provide redundant APIs. On the other hand, it's kind of weird to have only `get`.

==============

## `reserve_index` vs `reserve_len`

`reserve_len` is primarily motivated by BitvSet and VecMap, whose capacity semantics are largely
based around the largest index they have set, and not the number of elements they contain. This
design was chosen for its equivalence to `with_capacity`, as well as possible
future-proofing for adding it to other collections like `Vec` or `RingBuf`.

However one could instead opt for `reserve_index`, which are effectively the same method,
but with an off-by-one. That is, `reserve_len(x) == reserve_index(x - 1)`. This more closely
matches the intent (let me have index `7`), but has tricky off-by-one with `capacity`.

Alternatively `reserve_len` could just be called `reserve_capacity`.

==============

## RingBuf `as_slice`

Other designs for this usecase were considered:

```
/// Attempts to get a slice over all the elements in the RingBuf, but may instead
/// have to return two slices, in the case that the elements aren't contiguous.
pub fn as_slice(&'a self) -> RingBufSlice<'a, T>

enum RingBufSlice<'a, T> {
    Contiguous(&'a [T]),
    Split((&'a [T], &'a [T])),
}
```

```
/// Gets a slice over all the elements in the RingBuf. This may require shifting
/// all the elements to make this possible.
pub fn to_slice(&mut self) -> &[T]
```

The one settled on had the benefit of being the simplest. In particular, having the enum wasn't
very helpful, because most code would just create an empty slice anyway in the contiguous case
to avoid code-duplication.

# Unresolved questions

`reserve_index` vs `reserve_len` and `Ringbuf::as_slice` are the two major ones.

[part1]: https://github.com/rust-lang/rfcs/blob/master/text/0235-collections-conventions.md
