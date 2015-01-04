- Start Date: 2015-01-03
- RFC PR: [rust-lang/rfcs#546](https://github.com/rust-lang/rfcs/pull/546)
- Rust Issue: [rust-lang/rust#20497](https://github.com/rust-lang/rust/issues/20497)

# Summary

1. Remove the `Sized` default for the implicitly declared `Self`
   parameter on traits.
2. Make it "object unsafe" for a trait to inherit from `Sized`.

# Motivation

The primary motivation is to enable a trait object `SomeTrait` to
implement the trait `SomeTrait`. This was the design goal of enforcing
object safety, but there was a detail that was overlooked, which this
RFC aims to correct.

Secondary motivations include:

- More generality for traits, as they are applicable to DST.
- Eliminate the confusing and irregular `impl Trait for ?Sized`
  syntax.
- Sidestep questions about whether the `?Sized` default is inherited
  like other supertrait bounds that appear in a similar position.

This change has been implemented. Fallout within the standard library
was quite minimal, since the default only affects default method
implementations.

# Detailed design

Currently, all type parameters are `Sized` by default, including the
implicit `Self` parameter that is part of a trait definition. To avoid
the default `Sized` bound on `Self`, one declares a trait as follows
(this example uses the syntax accepted in [RFC 490] but not yet
implemented):

```rust
trait Foo for ?Sized { ... }
```

This syntax doesn't have any other precendent in the language. One
might expect to write:

```rust
trait Foo : ?Sized { ... }
```

However, placing `?Sized` in the supertrait listing raises awkward
questions regarding inheritance. Certainly, when experimenting with
this syntax early on, we found it very surprising that the `?Sized`
bound was "inherited" by subtraits. At the same time, it makes no
sense to inherit, since all that the `?Sized` notation is saying is
"do not add `Sized`", and you can't inherit the absence of a
thing. Having traits simply not inherit from `Sized` by default
sidesteps this problem altogether and avoids the need for a special
syntax to supress the (now absent) default.

Removing the default also has the benefit of making traits applicable
to more types by default. One particularly useful case is trait
objects. We are working towards a goal where the trait object for a
trait `Foo` always implements the trait `Foo`. Because the type `Foo`
is an unsized type, this is naturally not possible if `Foo` inherits
from `Sized` (since in that case every type that implements `Foo` must
also be `Sized`).

The impact of this change is minimal under the current rules. This is
because it only affects default method implementations. In any actual
impl, the `Self` type is bound to a specific type, and hence it known
whether or not that type is `Sized`. This change has been implemented
and hence the fallout can be seen on [this branch] (specifically,
[this commit] contains the fallout from the standard library). That
same branch also implements the changes needed so that every trait
object `Foo` implements the trait `Foo`.

[RFC 255]: https://github.com/rust-lang/rfcs/blob/master/text/0255-object-safety.md
[RFC 490]: https://github.com/rust-lang/rfcs/blob/master/text/0490-dst-syntax.md
[this branch]: https://github.com/nikomatsakis/rust/tree/impl-trait-for-trait-2
[this commit]: https://github.com/nikomatsakis/rust/commit/d08a08ab82031b6f935bdaf160a28d9520ded1ab

# Drawbacks

The `Self` parameter is inconsistent with other type parameters if we
adopt this RFC. We believe this is acceptable since it is
syntactically distinguished in other ways (for example, it is not
declared), and the benefits are substantial.

# Alternatives

- Leave `Self` as it is. The change to object safety must be made in
  any case, which would mean that for a trait object `Foo` to
  implement the trait `Foo`, it would have to be declared `trait Foo
  for Sized?`. Indeed, that would be necessary even to create a trait
  object `Foo`. This seems like an untenable burden, so adopting this
  design choice seems to imply reversing the decision that all trait
  objects implement their respective traits ([RFC 255]).
  
- Remove the `Sized` defaults altogether. This approach is purer, but
  the annotation burden is substantial. We continue to experiment in
  the hopes of finding an alternative to current blanket default, but
  without success thus far (beyond the idea of doing global
  inference).

# Unresolved questions

- None.
