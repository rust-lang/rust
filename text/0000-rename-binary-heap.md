- Start Date: 2015-01-13
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Rename the standard correction `BinaryHeap` to `BinHeap`, in order to follow the existing naming convention.

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

The abbreviation rules do seem unclear. Sometimes the first word is abbreviated, sometimes the last. However there are also cases where the names are not abbreviated. Such inconsistency is undesirable, as Rust should not give an impression as "the promising language that has strangely inconsistent naming conventions for its standard collections".

# Detailed design

An observation:

Given the current names, Rust actually *do* have consistent name abbreviation rules that are generally followed by its standard collections:

- Each word in the names must be shorter than five letters, or they should be abbreviated.
- When choosing abbreviations for each overly-long word, prefer commonly used ones.

There are four names worth mentioning: `Bitv`, `BitvSet`, `DList` and `BinaryHeap`.

- `Bitv`: This can be seen as short for `Bitvector`, not `BitVector`, so it actually confirms to the rules.
- `BitvSet`: Ditto.
- `DList`: This should be either `DList`, or `DoublyLinkedList`, as all the "middle grounds" feel unnatural. (DLList? DblList? DoublyList? DLinkList? ...) We don't want the full one because it is too long (and more importantly, we already use other abbreviated names), so `DList` is the best choice here.
- `BinaryHeap`: It seems that `BinHeap` can be a better name here, no reason to violate the rules.

Thus, this RFC proposes the following change:

**Rename `BinaryHeap` to `BinHeap`.**

# Drawbacks

- This is A breaking change to a standard collection that is already marked `stable`.
- There is no guarantee that all future additions to the standard collections will have names that look pretty under these abbreviation rules.

However `BinaryHeap` is only one collection, and a deprecation period can be provided if necessary, so the first drawback may not be a serious problem. Regarding the second one, this change at least isn't worse than the status quo.

# Alternatives

## A. Keep the status quo:

And Rust will have no consistent abbreviation rules for its standard collections' names.

## B. Rename all collections to their full names:

This will ensure maximum consistency, both now and in the future. However, a breaking change at this scale is undesirable at this stage, and `Vec` is so frequently used that it deserves an abbreviation. Then, if one collection has an abbreviated name, it is only natural for others to also have such names.

## C. Also rename `Bitv` to `BitVec`, and `BitvSet` to `BitVecSet`:

Some may argue that `BitVector` is the more common spelling, not `Bitvector`, so `Bitv` is not as good as `BitVec`. Therefore, `Bitv` and `BitvSet` should also be renamed.

The drawback: this alternative means more breaking changes than only renaming `BinaryHeap`.

# Unresolved questions

None.
