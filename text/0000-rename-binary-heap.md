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

Thus, this RFC proposes the following changes:

- Rename `std::collections::binary_heap::BinaryHeap` to `std::collections::bin_heap::BinHeap`. Change affected codes accordingly.
- If necessary, redefine `BinaryHeap` as an alias of `BinHeap` and mark it as deprecated. After a transition period, remove `BinaryHeap` completely. 

# Drawbacks

- This is A breaking change to a standard collection that is already marked `stable`.
- `DList` is left unchanged, but far from ideal. It just doesn't say much about what the type actually is.
- Future additions to the standard collections may be like `DList`/`DoublyLinkedList` in that no ideal abbreviations can be found. Such additions *will* make the standard collections' names less consistent. 

This solution can bring *some* consistency to the collections' names, but doing so may be sweeping the real problem under the rug. Still it is better than the status quo and requires the least amount of breaking changes.

# Alternatives

## A. Keep the status quo:

And Rust's standard collections will have no consistent name abbreviation rules. `DList` can be excused for being an exception (if only because all the alternatives are worse), but `BinaryHeap` cannot.

## B. Rename all collections with abbreviated names to their full names:

This will ensure maximum consistency, both now and in the future. As the referenced reddit comment (and discussions about this RFC) indicates, *Many* believe this to be the optimal solution.

However:

- A breaking change at this scale is undesirable at this stage.
- `Vec` is so frequently used that it deserves an abbreviation.
- If one collection has an abbreviated name, it is only natural for others to also have such names.
- Most abbreviated names are clear, `DList` is the exception, not the rule.

Still, using full and consistent names may be the right choice in the long run, especially considering that people tend to follow the naming conventions of the standard library, and it's very likely that there will be future additions to the standard collections, which may or may not have "abbreviation-friendly" names.

Also, if abbreviated names are truly needed, one can always write `type`. `Option` is not called `Opt` after all. Some may also argue that modern editors/IDEs make longer names less of an issue.

## C. Rename `BinaryHeap`, and also `Bitv` to `BitVec`, `BitvSet` to `BitVecSet`:

Some may argue that `BitVector` is the more common spelling, not `Bitvector`, so `Bitv` is not as good as `BitVec`. Therefore, `Bitv` and `BitvSet` should also be renamed alongside `BinaryHeap`.

The pros and cons of this alternative is similar to only renaming `BinaryHeap`, but with more conventional names and more breaking changes.

# Unresolved questions

None.
