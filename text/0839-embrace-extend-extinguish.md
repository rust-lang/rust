- Feature Name: embrace-extend-extinguish
- Start Date: 2015-02-13
- RFC PR: [rust-lang/rfcs#839](https://github.com/rust-lang/rfcs/pull/839)
- Rust Issue: [rust-lang/rust#25976](https://github.com/rust-lang/rust/issues/25976)

# Summary

Make all collections `impl<'a, T: Copy> Extend<&'a T>`.

This enables both `vec.extend(&[1, 2, 3])`, and `vec.extend(&hash_set_of_ints)`.
This partially covers the usecase of the awkward `Vec::push_all` with
literally no ergonomic loss, while leveraging established APIs.

# Motivation

Vec::push_all is kinda random and specific. Partially motivated by performance concerns,
but largely just "nice" to not have to do something like
`vec.extend([1, 2, 3].iter().cloned())`. The performance argument falls flat
(we *must* make iterators fast, and trusted_len should get us there). The ergonomics
argument is salient, though. Working with Plain Old Data types in Rust is super annoying
because generic APIs and semantics are tailored for non-Copy types.

Even with Extend upgraded to take IntoIterator, that won't work with &[Copy],
because a slice can't be moved out of. Collections would have to take `IntoIterator<&T>`,
and copy out of the reference. So, do exactly that.

As a bonus, this is more expressive than `push_all`, because you can feed in *any*
collection by-reference to clone the data out of it, not just slices.

# Detailed design

* For sequences and sets: `impl<'a, T: Copy> Extend<&'a T>`
* For maps: `impl<'a, K: Copy, V: Copy> Extend<(&'a K, &'a V)>`

e.g.

```rust
use std::iter::IntoIterator;

impl<'a, T: Copy> Extend<&'a T> for Vec<T> {
    fn extend<I: IntoIterator<Item=&'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().cloned())
    }
}


fn main() {
    let mut foo = vec![1];
    foo.extend(&[1, 2, 3, 4]);
    let bar = vec![1, 2, 3];
    foo.extend(&bar);
    foo.extend(bar.iter());

    println!("{:?}", foo);
}
```

# Drawbacks

* Mo' generics, mo' magic. How you gonna discover it?

* This creates a potentially confusing behaviour in a generic context.

Consider the following code:

```rust
fn feed<'a, X: Extend<&'a T>>(&'a self, buf: &mut X) {
    buf.extend(self.data.iter());
}
```

One would reasonably expect X to contain &T's, but with this
proposal it is possible that X now instead contains T's. It's not
clear that in "real" code that this would ever be a problem, though.
It may lead to novices accidentally by-passing ownership through
implicit copies.

It also may make inference fail in some other cases, as Extend would
not always be sufficient to determine the type of a `vec![]`.

* This design does not fully replace the push_all, as it takes `T: Clone`.

# Alternatives


## The Cloneian Candidate
This proposal is artifically restricting itself to `Copy` rather than full
`Clone` as a concession to the general Rustic philosophy of Clones being
explicit. Since this proposal is largely motivated by simple shuffling of
primitives, this is sufficient. Also, because `Copy: Clone`, it would be
backwards compatible to upgrade to `Clone` in the future if demand is
high enough.

## The New Method
It is theoretically plausible to add a new defaulted method to Extend called
`extend_cloned` that provides this functionality. This removes any concern of
accidental clones and makes inference totally work. However this design cannot
simultaneously support Sequences and Maps, as the signature for sequences would
mean Maps can only Copy through &(K, V), rather than (&K, &V). This would make
it impossible to copy-chain Maps through Extend.

## Why not FromIterator?

FromIterator could also be extended in the same manner, but this is less useful for
two reasons:

* FromIterator is always called by calling `collect`, and IntoIterator doesn't really
"work" right in `self` position.
* Introduces ambiguities in some cases. What is `let foo: Vec<_> = [1, 2, 3].iter().collect()`?

Of course, context might disambiguate in many cases, and
`let foo: Vec<i32> = [1, 2, 3].iter().collect()` might still be nicer than
`let foo: Vec<_> = [1, 2, 3].iter().cloned().collect()`.


# Unresolved questions

None.

