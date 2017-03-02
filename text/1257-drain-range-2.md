- Feature Name: drain-range
- Start Date: 2015-08-14
- RFC PR: [rust-lang/rfcs#1257](https://github.com/rust-lang/rfcs/pull/1257)
- Rust Issue: [rust-lang/rust#27711](https://github.com/rust-lang/rust/issues/27711)

# Summary

Implement `.drain(range)` and `.drain()` respectively as appropriate on collections.

# Motivation

The `drain` methods and their draining iterators serve to mass remove elements
from a collection, receiving them by value in an iterator, while the collection
keeps its allocation intact (if applicable).

The range parameterized variants of drain are a generalization of `drain`, to
affect just a subrange of the collection, for example removing just an index range
from a vector.

`drain` thus serves both to consume all or some elements from a collection without
consuming the collection itself. The ranged `drain` allows bulk removal of
elements, more efficently than any other safe API.

# Detailed design

- Implement `.drain(a..b)` where `a` and `b` are indices, for all
  collections that are sequences.
- Implement `.drain()` for other collections. This is just like `.drain(..)` would be
  (drain the whole collection).
- Ranged drain accepts all range types, currently .., a.., ..b, a..b,
  and drain will accept inclusive end ranges ("closed ranges") when they are implemented.
- Drain removes every element in the range.
- Drain returns an iterator that produces the removed items by value.
- Drain removes the whole range, regardless if you iterate the draining iterator
  or not.
- Drain preserves the collection's capacity where it is possible.

## Collections

`Vec` and `String` already have ranged drain, so they are complete.

`HashMap` and `HashSet` already have `.drain()`, so they are complete;
their elements have no meaningful order.

`BinaryHeap` already has `.drain()`, and just like its other iterators,
it promises no particular order. So this collection is already complete.

The following collections need updated implementations:

`VecDeque` should implement `.drain(range)` for index ranges, just like `Vec`
does.

`LinkedList` should implement `.drain(range)` for index ranges. Just
like the other seqences, this is a `O(n)` operation, and `LinkedList` already
has other indexed methods (`.split_off()`).

## `BTreeMap` and `BTreeSet`

`BTreeMap` already has a ranged iterator, `.range(a, b)`, and `drain` for
`BTreeMap` and `BTreeSet` should have arguments completely consistent the range
method. This will be addressed separately.

## Stabilization

The following can be stabilized as they are:

- `HashMap::drain`
- `HashSet::drain`
- `BinaryHeap::drain`

The following can be stabilized, but their argument's trait is not stable:

- `Vec::drain`
- `String::drain`

The following will be heading towards stabilization after changes:

- `VecDeque::drain`

# Drawbacks

- Collections disagree on if they are drained with a range (`Vec`) or not (`HashMap`)
- No trait for the drain method.

# Alternatives

- Use a trait for the drain method and let all collections implement it. This
  will force all collections to use a single parameter (a range) for the drain
  method.

- Provide `.splice(range, iterator)` for `Vec` instead of `.drain(range)`:

  ```rust
  fn splice<R, I>(&mut self, range: R, iter: I) -> Splice<T>
      where R: RangeArgument, I: IntoIterator<T>
  ```

  if the method `.splice()` would both return an iterator of the replaced elements,
  and consume an iterator (of arbitrary length) to replace the removed range, then
  it includes drain's tasks.

- RFC #574 proposed accepting either a single index (single key for maps)
  or a range for ranged drain, so an alternative would be to do that. The
  single index case is however out of place, and writing a range that spans
  a single index is easy.

- Use the name `.remove_range(a..b)` instead of `.drain(a..b)`. Since the method
  has two simultaneous roles, removing a range and yielding a range as an iterator,
  either role could guide the name.
  This alternative name was not very popular with the rust developers I asked
  (but they are already used to what `drain` means in rust context).

- Provide `.drain()` without arguments and separate range drain into a separate
  method name, implemented in addition to `drain` where applicable.

- Do not support closed ranges in `drain`.

- `BinaryHeap::drain` could drain the heap in sorted order. The primary proposal
  is arbitrary order, to match preexisting `BinaryHeap` iterators.

# Unresolved questions

- Concrete shape of the `BTreeMap` API is not resolved here
- Will closed ranges be used for the `drain` API?
