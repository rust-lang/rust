- Start Date: 2015-01-13
- RFC PR: https://github.com/rust-lang/rfcs/pull/580
- Rust Issue: https://github.com/rust-lang/rust/issues/22479

# Summary

Rename (maybe one of) the standard collections, so as to make the names more consistent. Currently, among all the alternatives, renaming `BinaryHeap` to `BinHeap` is the slightly preferred solution.

# Motivation

In [this comment](http://www.reddit.com/r/programming/comments/2rvoha/announcing_rust_100_alpha/cnk31hf) in the Rust 1.0.0-alpha announcement thread in /r/programming, it was pointed out that Rust's std collections had inconsistent names. Particularly, the abbreviation rules of the names seemed unclear.

The current collection names (and their longer versions) are:

* `Vec` -> `Vector`
* `BTreeMap`
* `BTreeSet`
* `BinaryHeap`
* `Bitv` -> `BitVec` -> `BitVector`
* `BitvSet` -> `BitVecSet` -> `BitVectorSet`
* `DList` -> `DoublyLinkedList`
* `HashMap`
* `HashSet`
* `RingBuf` -> `RingBuffer`
* `VecMap` -> `VectorMap`

The abbreviation rules do seem unclear. Sometimes the first word is abbreviated, sometimes the last. However there are also cases where the names are not abbreviated. `Bitv`, `BitvSet` and `DList` seem strange on first glance. Such inconsistencies are undesirable, as Rust should not give an impression as "the promising language that has strangely inconsistent naming conventions for its standard collections".

Also, it should be noted that traditionally *ring buffers* have fixed sizes, but Rust's `RingBuf` does not. So it is preferable to rename it to something clearer, in order to avoid incorrect assumptions and surprises.

# Detailed design

First some general naming rules should be established.

1. At least maintain module level consistency when abbreviations are concerned.
2. Prefer commonly used abbreviations.
3. When in doubt, prefer full names to abbreviated ones.
4. Don't be dogmatic.

And the new names:

* `Vec`
* `BTreeMap`
* `BTreeSet`
* `BinaryHeap`
* `Bitv` -> `BitVec`
* `BitvSet` -> `BitSet`
* `DList` -> `LinkedList`
* `HashMap`
* `HashSet`
* `RingBuf` -> `VecDeque`
* `VecMap`

The following changes should be made:

- Rename `Bitv`, `BitvSet`, `DList` and `RingBuf`. Change affected codes accordingly.
- If necessary, redefine the original names as aliases of the new names, and mark them as deprecated. After a transition period, remove the original names completely.

## Why prefer full names when in doubt?

The naming rules should apply not only to standard collections, but also to other codes. It is (comparatively) easier to maintain a higher level of naming consistency by preferring full names to abbreviated ones *when in doubt*. Because given a full name, there are possibly many abbreviated forms to choose from. Which one should be chosen and why? It is hard to write down guidelines for that.

For example, the name `BinaryBuffer` has at least three convincing abbreviated forms: `BinBuffer`/`BinaryBuf`/`BinBuf`. Which one would be the most preferred? Hard to say. But it is clear that the full name `BinaryBuffer` is not a bad name.

However, if there *is* a convincing reason, one should not hesitate using abbreviated names. A series of names like `BinBuffer/OctBuffer/HexBuffer` is very natural. Also, few would think that `AtomicallyReferenceCounted`, the full name of `Arc`, is a good type name.

## Advantages of the new names:

- `Vec`: The name of the most frequently used Rust collection is left unchanged (and by extension `VecMap`), so the scope of the changes are greatly reduced. `Vec` is an exception to the "prefer full names" rule because it is *the* collection in Rust.
- `BitVec`: `Bitv` is a very unusual abbreviation of `BitVector`, but `BitVec` is a good one given `Vector` is shortened to `Vec`.
- `BitSet`: Technically, `BitSet` is a synonym of `BitVec(tor)`, but it has `Set` in its name and can be interpreted as a set-like "view" into the underlying bit array/vector, so `BitSet` is a good name. No need to have an additional `v`.
- `LinkedList`: `DList` doesn't say much about what it actually is. `LinkedList` is not too long (like `DoublyLinkedList`) and it being a doubly-linked list follows Java/C#'s traditions.
- `VecDeque`: This name exposes some implementation details and signifies its "interface" just like `HashSet`, and it doesn't have the "fixed-size" connotation that `RingBuf` has. Also, `Deque` is commonly preferred to `DoubleEndedQueue`, it is clear that the former should be chosen.

# Drawbacks

- There will be breaking changes to standard collections that are already marked `stable`.

# Alternatives

## A. Keep the status quo:

And Rust's standard collections will have some strange names and no consistent naming rules.

## B. Also rename `Vec` to `Vector`:

And by extension, `Bitv` to `BitVector` and `VecMap` to `VectorMap`.

This means breaking changes at a larger scale. Given that `Vec` is *the* collection of Rust, we can have an exception here.

## C. Rename `DList` to `DLinkedList`, not `LinkedList`:

It is clearer, but also inconsistent with the other names by having a single-lettered abbreviation of `Doubly`. As Java/C# also have doubly-linked `LinkedList`, it is not necessary to use the additional `D`.

## D. Also rename `BinaryHeap` to `BinHeap`.

`BinHeap` can also mean `BinomialHeap`, so `BinaryHeap` is the better name here.

## E. Rename `RingBuf` to `RingBuffer`, or do not rename `RingBuf` at all.

Doing so would fail to stop people from making the incorrect assumption that Rust's `RingBuf`s have fixed sizes.

# Unresolved questions

None.
