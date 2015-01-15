- Start Date: 2015-01-13
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

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

# Detailed design

First some general naming rules should be established.

1. Prefer commonly used names.
2. Prefer full names when full names and abbreviated names are almost equally elegant.

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
* `RingBuf` -> `RingBuffer`
* `VecMap`

The following changes should be made:

- Rename `Bitv`, `BitvSet`, `DList` and `RingBuf`. Change affected codes accordingly.
- If necessary, redefine the original names as aliases of the new names, and mark them as deprecated. After a transition period, remove the original names completely.

## Why prefer full names when full names and abbreviated ones are almost equally elegant?

The naming rules should apply not only to standard collections, but also to other codes. It is (comparatively) easier to maintain a higher level of naming consistency by preferring full names to abbreviated ones *when in doubt*. Because given a full name, there are possibly many abbreviated forms to choose from. Which should be chosen and why? It is hard to write down guideline for that.

For example, a name `BinaryBuffer` have at least three convincing abbreviated forms: `BinBuffer`/`BinaryBuf`/`BinBuf`. Which one would be the most preferred? Hard to say. But it is clear that the full name `BinaryBuffer` is not a bad name.

However, if there *is* a convincing reason, one should not hesitate using abbreviated names. A series of names like `BinBuffer/OctBuffer/HexBuffer` is very natural. Also, few would think the full name of `Arc` is a good type name.

## Advantages of the new names:

- `Vec`: The name of the most frequently used Rust collection is left unchanged (and by extension `VecMap`), so the scope of the changes are greatly reduced. `Vec` is an exception to the rule because it is *the* collection in Rust.
- `BitVec`: `Bitv` is a very unusual abbreviation of `BitVector`, but `BitVec` is a good one given `Vector` is shortened to `Vec`.
- `BitSet`: Technically, `BitSet` is a synonym of `BitVec(tor)`, but it has `Set` in its name and can be interpreted as a set-like "view" into the underlying bit array/vector, so `BitSet` is a good name. No need to have an additional `v`.
- `LinkedList`: `DList` doesn't say much about what it actually is. `LinkedList` is not too long (like `DoublyLinkedList`) and it being a doubly-linked list follows Java/C#'s traditions.
- `RingBuffer`: `RingBuf` is a good name, but `RingBuffer` is good too. No reason to violate the rule here.

# Drawbacks

- Preferring full names may result in people naming things with overly-long names that are hard to write and more importantly, read.
- There will be breaking changes to standard collections that are already marked `stable`.

# Alternatives

## A. Keep the status quo:

And Rust's standard collections will have some strange names and no consistent naming rules.

## B. Also rename `Vec` to `Vector`:

And by extension, `Bitv` to `BitVector` and `VecMap` to `VectorMap`.

This means breaking changes at a much larger scale. Undesirable at this stage.

## C. Rename `DList` to `DLinkedList`, not `LinkedList`:

It is clearer, but also inconsistent with the other names by having a single-lettered abbreviation of `Doubly`. As Java/C# also have doubly-linked `LinkedList`, it is not necessary to use the additional `D`.

## D. Instead of renaming `RingBuf` to `RingBuffer`, rename `BinaryHeap` to `BinHeap`.

Or, reversing the second rule: prefer abbreviated names to full ones when in doubt.

This has the advantage of encouraging succinct names, but everyone has his/her own preferences of how to abbreviate things. Naming consistency will suffer. Whether this is a problem is also a quite subjective matter.

# Unresolved questions

None.
