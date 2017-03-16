- Feature Name: `sort_unstable`
- Start Date: 2017-02-03
- RFC PR: [rust-lang/rfcs#1884](https://github.com/rust-lang/rfcs/pull/1884)
- Rust Issue: [rust-lang/rust#40585](https://github.com/rust-lang/rust/issues/40585)

# Summary
[summary]: #summary

Add an unstable sort to libcore.

# Motivation
[motivation]: #motivation

At the moment, the only sort function we have in libstd is `slice::sort`. It is stable,
allocates additional memory, and is unavailable in `#![no_std]` environments.

The sort function is stable, which is a good but conservative default. However,
stability is rarely a required property in practice, and some other characteristics
of sort algorithms like higher performance or lower memory overhead are often more
desirable.

Having a performant, non-allocating unstable sort function in libcore would cover those
needs. At the moment Rust is not offering this solution as a built-in (only crates), which
is unusual for a systems programming language.

**Q: What is stability?**<br>
A: A sort function is stable if it doesn't reorder equal elements. For example:
```rust
let mut orig = vec![(0, 5), (0, 4)];
let mut v = orig.clone();

// Stable sort preserves the original order of equal elements.
v.sort_by_key(|p| p.0);
assert!(orig == v); // OK!

/// Unstable sort may or may not preserve the original order.
v.sort_unstable_by_key(|p| p.0);
assert!(orig == v); // MAY FAIL!
```

**Q: When is stability useful?**<br>
A: Not very often. A typical example is sorting columns in interactive GUI tables.
E.g. you want to have rows sorted by column X while breaking ties by column Y, so you
first click on column Y and then click on column X. This is a use case where stability
is important.

**Q: Can stable sort be performed using unstable sort?**<br>
A: Yes. If we transform `[T]` into `[(T, usize)]` by pairing every element with it's
index, then perform unstable sort, and finally remove indices, the result will be
equivalent to stable sort.

**Q: Why is `slice::sort` stable?**<br>
A: Because stability is a good default. A programmer might call a sort function
without checking in the documentation whether it is stable or unstable. It is very
intuitive to assume stability, so having `slice::sort` perform unstable sorting might
cause unpleasant surprises.
See this [story](https://medium.com/@cocotutch/a-swift-sorting-problem-e0ebfc4e46d4#.yfvsgjozx)
for an example.

**Q: Why does `slice::sort` allocate?**<br>
A: It is possible to implement a non-allocating stable sort, but it would be
considerably slower.

**Q: Why is `slice::sort` not compatible with `#![no_std]`?**<br>
A: Because it allocates additional memory.

**Q: How much faster can unstable sort be?**<br>
A: Sorting 10M 64-bit integers using [pdqsort][stjepang-pdqsort] (an
unstable sort implementation) is **45% faster** than using `slice::sort`.
Detailed benchmarks are [here](https://github.com/stjepang/pdqsort#extensive-benchmarks).

**Q: Can unstable sort benefit from allocation?**<br>
A: Generally, no. There is no fundamental property in computer science saying so,
but this has always been true in practice. Zero-allocation and instability go
hand in hand.

# Detailed design
[design]: #detailed-design

The API will consist of three functions that mirror the current sort in libstd:

1. `core::slice::sort_unstable`
2. `core::slice::sort_unstable_by`
3. `core::slice::sort_unstable_by_key`

By contrast, C++ has functions `std::sort` and `std::stable_sort`, where the
defaults are set up the other way around.

### Interface

```rust
pub trait SliceExt {
    type Item;

    // ...

    fn sort_unstable(&mut self)
        where Self::Item: Ord;

    fn sort_unstable_by<F>(&mut self, compare: F)
        where F: FnMut(&Self::Item, &Self::Item) -> Ordering;

    fn sort_unstable_by_key<B, F>(&mut self, mut f: F)
        where F: FnMut(&Self::Item) -> B,
              B: Ord;
}
```

### Examples

```rust
let mut v = [-5i32, 4, 1, -3, 2];

v.sort_unstable();
assert!(v == [-5, -3, 1, 2, 4]);

v.sort_unstable_by(|a, b| b.cmp(a));
assert!(v == [4, 2, 1, -3, -5]);

v.sort_unstable_by_key(|k| k.abs());
assert!(v == [1, 2, -3, 4, -5]);
```

### Implementation

Proposed implementaton is available in the [pdqsort][stjepang-pdqsort] crate.

**Q: Why choose this particular sort algorithm?**<br>
A: First, let's analyse what unstable sort algorithms other languages use:

* C: quicksort
* C++: introsort
* D: introsort
* Swift: introsort
* Go: introsort
* Crystal: introsort
* Java: dual-pivot quicksort

The most popular sort is definitely introsort. Introsort is an implementation
of quicksort that limits recursion depth. As soon as depth exceeds `2 * log(n)`,
it switches to heapsort in order to guarantee `O(n log n)` worst-case. This
method combines the best of both worlds: great average performance of
quicksort with great worst-case performance of heapsort.

Java (talking about `Arrays.sort`, not `Collections.sort`) uses dual-pivot
quicksort. It is an improvement of quicksort that chooses two pivots for finer
grained partitioning, offering better performance in practice.

A recent improvement of introsort is [pattern-defeating quicksort][orlp-pdqsort],
which is substantially faster in common cases. One of the key tricks pdqsort
uses is block partitioning described in the [BlockQuicksort][blockquicksort] paper.
This algorithm still hasn't been built into in any programming language's
standard library, but there are plans to include it into some C++ implementations.

Among all these, pdqsort is the clear winner. Some benchmarks are available
[here](https://github.com/stjepang/pdqsort#a-simple-benchmark).

**Q: Is `slice::sort` ever faster than pdqsort?**<br>
A: Yes, there are a few cases where it is faster. For example, if the slice
consists of several pre-sorted sequences concatenated one after another, then
`slice::sort` will most probably be faster. Another case is when using costly
comparison functions, e.g. when sorting strings. `slice::sort` optimizes the
number of comparisons very well, while pdqsort optimizes for fewer writes to
memory at expense of slightly larger number of comparisons. But other than
that, `slice::sort` should be generally slower than pdqsort.

**Q: What about radix sort?**<br>
A: Radix sort is usually blind to patterns in slices. It treats totally random
and partially sorted the same way. It is probably possible to improve it
by combining it with some other techniques, but it's not trivial. Moreover,
radix sort is incompatible with comparison-based sorting, which makes it
an awkward choice for a general-purpose API. On top of all this, it's
not even that much faster than pdqsort anyway.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

Stability is a confusing and loaded term. Function `slice::sort_unstable` might be
misunderstood as a function that has unstable API. That said, there is no
less confusing alternative to "unstable sorting". Documentation should
clearly state what "stable" and "unstable" mean.

`slice::sort_unstable` will be mentioned in the documentation for `slice::sort`
as a faster non-allocating alternative. The documentation for
`slice::sort_unstable` must also clearly state that it guarantees no allocation.

# Drawbacks
[drawbacks]: #drawbacks

The amount of code for sort algorithms will grow, and there will be more code
to review.

It might be surprising to discover cases where `slice::sort` is faster than
`slice::sort_unstable`. However, these peculiarities can be explained in
documentation.

# Alternatives
[alternatives]: #alternatives

Unstable sorting is indistinguishable from stable sorting when sorting
primitive integers. It's possible to specialize `slice::sort` to fall back
to `slice::sort_unstable`. This would improve performance for primitive integers in
most cases, but patching cases type by type with different algorithms makes
performance more inconsistent and less predictable.

Unstable sort guarantees no allocation. Instead of naming it `slice::sort_unstable`,
it could also be named `slice::sort_noalloc` or `slice::sort_unstable_noalloc`.
This may slightly improve clarity, but feels much more awkward.

Unstable sort can also be provided as a standalone crate instead of
within the standard library. However, every other systems programming language
has a fast unstable sort in standard library, so why shouldn't Rust, too?

# Unresolved questions
[unresolved]: #unresolved-questions

None.

[orlp-pdqsort]: https://github.com/orlp/pdqsort
[stjepang-pdqsort]: https://github.com/stjepang/pdqsort
[blockquicksort]: http://drops.dagstuhl.de/opus/volltexte/2016/6389/pdf/LIPIcs-ESA-2016-38.pdf
