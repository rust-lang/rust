- Start Date: 2014-06-01
- RFC PR: [rust-lang/rfcs#100](https://github.com/rust-lang/rfcs/pull/100)
- Rust Issue: [rust-lang/rust#14987](https://github.com/rust-lang/rust/issues/14987)

# Summary

Add a `partial_cmp` method to `PartialOrd`, analagous to `cmp` in `Ord`.

# Motivation

The `Ord::cmp` method is useful when working with ordered values. When the
exact ordering relationship between two values is required, `cmp` is both
potentially more efficient than computing both `a > b` and then `a < b` and
makes the code clearer as well.

I feel that in the case of partial orderings, an equivalent to `cmp` is even
more important. I've found that it's very easy to accidentally make assumptions
that only hold true in the total order case (for example `!(a < b) => a >= b`).
Explicitly matching against the possible results of the comparison helps keep
these assumptions from creeping in.

In addition, the current default implementation setup is a bit strange, as
implementations in the *partial* equality trait assume *total* equality. This
currently makes it easier to incorrectly implement `PartialOrd` for types that
do not have a total ordering, and if `PartialOrd` is separated from `Ord` in a
way similar to [this](https://gist.github.com/alexcrichton/10945968) proposal,
the default implementations for `PartialOrd` will need to be removed and an
implementation of the trait will require four repetitive implementations of
the required methods.

# Detailed design

Add a method to `PartialOrd`, changing the default implementations of the other
methods:
```rust
pub trait PartialOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering>;

    fn lt(&self, other: &Self) -> bool {
        match self.partial_cmp(other) {
            Some(Less) => true,
            _ => false,
        }
    }

    le(&self, other: &Self) -> bool {
        match self.partial_cmp(other) {
            Some(Less) | Some(Equal) => true,
            _ => false,
        }
    }

    fn gt(&self, other: &Self) -> bool {
        match self.partial_cmp(other) {
            Some(Greater) => true,
            _ => false,
        }
    }

    ge(&self, other: &Self) -> bool {
        match self.partial_cmp(other) {
            Some(Greater) | Some(Equal) => true,
            _ => false,
        }
    }
}
```

Since almost all ordered types have a total ordering, the implementation of
`partial_cmp` is trivial in most cases:
```rust
impl PartialOrd for Foo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
```
This can be done automatically if/when RFC #48 or something like it is accepted
and implemented.

# Drawbacks

This does add some complexity to `PartialOrd`. In addition, the more commonly
used methods (`lt`, etc) may become more expensive than they would normally be
if their implementations call into `partial_ord`.

# Alternatives

We could invert the default implementations and have a default implementation
of `partial_cmp` in terms of `lt` and `gt`. This may slightly simplify things
in current Rust, but it makes the default implementation less efficient than it
should be. It would also require more work to implement `PartialOrd` once the
currently planned `cmp` reform has finished as noted above.

`partial_cmp` could just be called `cmp`, but it seems like UFCS would need to
be implemented first for that to be workrable.

# Unresolved questions

We may want to add something similar to `PartialEq` as well. I don't know what
it would be called, though (maybe `partial_eq`?):
```rust
// I don't feel great about these variant names, but `Equal` is already taken
// by `Ordering` which is in the same module.
pub enum Equality {
    AreEqual,
    AreUnequal,
}

pub trait PartialEq {
    fn partial_eq(&self, other: &Self) -> Option<Equality>;

    fn eq(&self, other: &Self) -> bool {
        match self.partial_eq(other) {
            Some(AreEqual) => true,
            _ => false,
        }
    }

    fn neq(&self, other: &Self) -> bool {
        match self.partial_eq(other) {
            Some(AreUnequal) => true,
            _ => false,
        }
    }
}
```
