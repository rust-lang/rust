- Start Date: 2014-11-29
- RFC PR: [rust-lang/rfcs#459](https://github.com/rust-lang/rfcs/pull/459)
- Rust Issue: [rust-lang/rust#19390](https://github.com/rust-lang/rust/issues/19390)

# Summary

Disallow type/lifetime parameter shadowing.

# Motivation

Today we allow type and lifetime parameters to be shadowed. This is a
common source of bugs as well as confusing errors. An example of such a confusing case is:

```rust
struct Foo<'a> {
    x: &'a int
}

impl<'a> Foo<'a> {
    fn set<'a>(&mut self, v: &'a int) {
        self.x = v;
    }
}

fn main() { }
```

In this example, the lifetime parameter `'a` is shadowed on the method, leading to two
logically distinct lifetime parameters with the same name. This then leads to the error
message:

    mismatched types: expected `&'a int`, found `&'a int` (lifetime mismatch)

which is obviously completely unhelpful.

Similar errors can occur with type parameters:

```rust
struct Foo<T> {
    x: T
}

impl<T> Foo<T> {
    fn set<T>(&mut self, v: T) {
        self.x = v;
    }
}

fn main() { }
```

Compiling this program yields:

    mismatched types: expected `T`, found `T` (expected type parameter, found a different type parameter)

Here the error message was improved by [a recent PR][pr], but this is
still a somewhat confusing situation.

Anecdotally, this kind of accidental shadowing is fairly frequent
occurrence.  It recently arose on [this discuss thread][dt], for
example.

[dt]: http://discuss.rust-lang.org/t/confused-by-lifetime-error-messages-tell-me-about-it/358/41?u=nikomatsakis
[pr]: https://github.com/rust-lang/rust/pull/18264

# Detailed design

Disallow shadowed type/lifetime parameter declarations. An error would
be reported by the resolve/resolve-lifetime passes in the compiler and
hence fairly early in the pipeline.

# Drawbacks

We otherwise allow shadowing, so it is inconsistent.

# Alternatives

We could use a lint instead. However, we'd want to ensure that the
lint error messages were printed *before* type-checking begins. We
could do this, perhaps, by running the lint printing pass multiple
times. This might be useful in any case as the placement of lints in
the compiler pipeline has proven problematic before.

We could also attempt to improve the error messages. Doing so for
lifetimes is definitely important in any case, but also somewhat
tricky due to the extensive inference. It is usually easier and more
reliable to help avoid the error in the first place.

# Unresolved questions

None.
