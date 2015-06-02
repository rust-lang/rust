- Feature Name: remove-static-assert
- Start Date: 2015-04-28        
- RFC PR: https://github.com/rust-lang/rfcs/pull/1096
- Rust Issue: https://github.com/rust-lang/rust/pull/24910

# Summary

Remove the `static_assert` feature.

# Motivation

To recap, `static_assert` looks like this:

```rust
#![feature(static_assert)]
#[static_assert]
static asssertion: bool = true;
```

If `assertion` is `false` instead, this fails to compile:

```text
error: static assertion failed
static asssertion: bool = false;
                          ^~~~~
```

If you don’t have the `feature` flag, you get another interesting error:

```text
error: `#[static_assert]` is an experimental feature, and has a poor API
```

Throughout its life, `static_assert` has been... weird. Graydon suggested it
[in May of 2013][suggest], and it was
[implemented][https://github.com/rust-lang/rust/pull/6670] shortly after.
[Another issue][issue] was created to give it a ‘better interface’. Here’s why:

> The biggest problem with it is you need a static variable with a name, that
> goes through trans and ends up in the object file.

In other words, `assertion` above ends up as a symbol in the final output. Not
something you’d usually expect from some kind of static assertion.

[suggest]: https://github.com/rust-lang/rust/issues/6568
[issue]: https://github.com/rust-lang/rust/issues/6676

So why not improve `static_assert`? With compile time function evaluation, the
idea of a ‘static assertion’ doesn’t need to have language semantics. Either
`const` functions or full-blown CTFE is a useful feature in its own right that
we’ve said we want in Rust. In light of it being eventually added,
`static_assert` doesn’t make sense any more.

`static_assert` isn’t used by the compiler at all.

# Detailed design

Remove `static_assert`. [Implementation submitted here][here].

[here]: https://github.com/rust-lang/rust/pull/24910

# Drawbacks

Why should we *not* do this?

# Alternatives

This feature is pretty binary: we either remove it, or we don’t. We could keep the feature,
but build out some sort of alternate version that’s not as weird.

# Unresolved questions

None with the design, only “should we do this?”
