- Feature Name: fundamental_attribute
- Start Date: 2015-03-27
- RFC PR: https://github.com/rust-lang/rfcs/pull/1023
- Rust Issue: https://github.com/rust-lang/rust/issues/23918

## Summary

This RFC proposes two rule changes:

1. Modify the orphan rules so that impls of remote traits require a
   local type that is either a struct/enum/trait defined in the
   current crate `LT = LocalTypeConstructor<...>` or a reference to a
   local type `LT = ... | &LT | &mut LT`.
2. Restrict negative reasoning so it too obeys the orphan rules.
3. Introduce an unstable `#[fundamental]` attribute that can be used
   to extend the above rules in select cases (details below).

## Motivation

The current orphan rules are oriented around allowing as many remote
traits as possible. As so often happens, giving power to one party (in
this case, downstream crates) turns out to be taking power away from
another (in this case, upstream crates). The problem is that due to
coherence, the ability to define impls is a zero-sum game: every impl
that is legal to add in a child crate is also an impl that a parent
crate cannot add without fear of breaking downstream crates. A
detailed look at these problems is
[presented here](https://gist.github.com/nikomatsakis/bbe6821b9e79dd3eb477);
this RFC doesn't go over the problems in detail, but will reproduce
some of the examples found in that document.

This RFC proposes a shift that attempts to strike a balance between
the needs of downstream and upstream crates. In particular, we wish to
preserve the ability of upstream crates to add impls to traits that
they define, while still allowing downstream creates to define the
sorts of impls they need.

While exploring the problem, we found that in practice remote impls
almost always are tied to a local type or a reference to a local
type. For example, here are some impls from the definition of `Vec`:

```rust
// tied to Vec<T>
impl<T> Send for Vec<T>
    where T: Send

// tied to &Vec<T>
impl<'a,T> IntoIterator for &'a Vec<T>
```

On this basis, we propose that we limit remote impls to require that
they include a type either defined in the current crate or a reference
to a type defined in the current crate. This is more restrictive than
the current definition, which merely requires a local type appear
*somewhere*. So, for example, under this definition `MyType` and
`&MyType` would be considered local, but `Box<MyType>`,
`Option<MyType>`, and `(MyType, i32)` would not.

Furthermore, we limit the use of *negative reasoning* to obey the
orphan rules. That is, just as a crate cannot define an impl `Type:
Trait` unless `Type` or `Trait` is local, it cannot rely that `Type:
!Trait` holds unless `Type` or `Trait` is local.

Together, these two changes cause very little code breakage while
retaining a lot of freedom to add impls in a backwards compatible
fashion. However, they are not quite sufficient to compile all the
most popular cargo crates (though they almost succeed). Therefore, we
propose an simple, unstable attribute `#[fundamental]` (described
below) that can be used to extend the system to accommodate some
additional patterns and types. This attribute is unstable because it
is not clear whether it will prove to be adequate or need to be
generalized; this part of the design can be considered somewhat
incomplete, and we expect to finalize it based on what we observe
after the 1.0 release.

### Practical effect

#### Effect on parent crates

When you first define a trait, you must also decide whether that trait
should have (a) a blanket impls for all `T` and (b) any blanket impls
over references. These blanket impls cannot be added later without a
major vesion bump, for fear of breaking downstream clients.

Here are some examples of the kinds of blanket impls that must be added
right away:

```rust
impl<T:Foo> Bar for T { }
impl<'a,T:Bar> Bar for &'a T { }
```

#### Effect on child crates

Under the base rules, child crates are limited to impls that use local
types or references to local types. They are also prevented from
relying on the fact that `Type: !Trait` unless either `Type` or
`Trait` is local. This turns out to be have very little impact.

In compiling the libstd facade and librustc, exactly two impls were
found to be illegal, both of which followed the same pattern:

```rust
struct LinkedListEntry<'a> {
    data: i32,
    next: Option<&'a LinkedListEntry>
}

impl<'a> Iterator for Option<&'a LinkedListEntry> {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        if let Some(ptr) = *self {
            *self = Some(ptr.next);
            Some(ptr.data)
        } else {
            None
        }
    }
}
```

The problem here is that `Option<&LinkedListEntry>` is no longer
considered a local type. A similar restriction would be that one
cannot define an impl over `Box<LinkedListEntry>`; but this was not
observed in practice.

Both of these restrictions can be overcome by using a new type.  For
example, the code above could be changed so that instead of writing
the impl for `Option<&LinkedListEntry>`, we define a type `LinkedList`
that wraps the option and implement on that:

```rust
struct LinkedListEntry<'a> {
    data: i32,
    next: LinkedList<'a>
}

struct LinkedList<'a> {
    data: Option<&'a LinkedListEntry>
}

impl<'a> Iterator for LinkedList<'a> {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        if let Some(ptr) = self.data {
            *self = Some(ptr.next);
            Some(ptr.data)
        } else {
            None
        }
    }
}
```

#### Errors from cargo and the fundamental attribute

We also applied our prototype to all the "Most Downloaded" cargo
crates as well as the `iron` crate. That exercise uncovered a few
patterns that the simple rules presented thus far can't handle.

The first is that it is common to implement traits over boxed trait
objects. For example, the `error` crate defines an impl:

- `impl<E: Error> FromError<E> for Box<Error>`

Here, `Error` is a local trait defined in `error`, but `FromError` is
the trait from `libstd`. This impl would be illegal because
`Box<Error>` is not considered local as `Box` is not local.

The second is that it is common to use `FnMut` in blanket impls,
similar to how the `Pattern` trait in `libstd` works. The `regex` crate
in particular has the following impls:

- `impl<'t> Replacer for &'t str`
- `impl<F> Replacer for F where F: FnMut(&Captures) -> String`
- these are in conflict because this requires that `&str: !FnMut`, and
  neither `&str` nor `FnMut` are local to `regex`

Given that overloading over closures is likely to be a common request,
and that the `Fn` traits are well-known, core traits tied to the call
operator, it seems reasonable to say that implementing a `Fn` trait is
itself a breaking change. (This is not to suggest that there is
something *fundamental* about the `Fn` traits that distinguish them
from all other traits; just that if the goal is to have rules that
users can easily remember, saying that implememting a core operator
trait is a breaking change may be a reasonable rule, and it enables
useful patterns to boot -- patterns that are baked into the libstd
APIs.)

To accommodate these cases (and future cases we will no doubt
encounter), this RFC proposes an unstable attribute
`#[fundamental]`. `#[fundamental]` can be applied to types and traits
with the following meaning:

- A `#[fundamental]` type `Foo` is one where implementing a blanket
  impl over `Foo` is a breaking change. As described, `&` and `&mut` are
  fundamental. This attribute would be applied to `Box`, making `Box`
  behave the same as `&` and `&mut` with respect to coherence.
- A `#[fundamental]` trait `Foo` is one where adding an impl of `Foo`
  for an existing type is a breaking change. For now, the `Fn` traits
  and `Sized` would be marked fundamental, though we may want to
  extend this set to all operators or some other
  more-easily-remembered set.

The `#[fundamental]` attribute is intended to be a kind of "minimal
commitment" that still permits the most important impl patterns we see
in the wild. Because it is unstable, it can only be used within libstd
for now. We are eventually committed to finding some way to
accommodate the patterns above -- which could be as simple as
stabilizing `#[fundamental]` (or, indeed, reverting this RFC
altogether). It could also be a more general mechanism that lets users
specify more precisely what kind of impls are reserved for future
expansion and which are not.

## Detailed Design

### Proposed orphan rules

Given an impl `impl<P1...Pn> Trait<T1...Tn> for T0`, either `Trait`
must be local to the current crate, or:

1. At least one type must meet the `LT` pattern defined above. Let
   `Ti` be the first such type.
2. No type parameters `P1...Pn` may appear in the type parameters that
   precede `Ti` (that is, `Tj` where `j < i`).

### Type locality and negative reasoning

Currently the overlap check employs negative reasoning to segregate
blanket impls from other impls. For example, the following pair of
impls would be legal only if `MyType<U>: !Copy` for all `U` (the
notation `Type: !Trait` is borrowed from [RFC 586][586]):

```rust
impl<T:Copy> Clone for T {..}
impl<U> Clone for MyType<U> {..}
```

[586]: https://github.com/rust-lang/rfcs/pull/586

This proposal places limits on negative reasoning based on the orphan
rules. Specifically, we cannot conclude that a proposition like `T0:
!Trait<T1..Tn>` holds unless `T0: Trait<T1..Tn>` meets the orphan
rules as defined in the previous section.

In practice this means that, by default, you can only assume negative
things about traits and types defined in your current crate, since
those are under your direct control. This permits parent crates to add
any impls except for blanket impls over `T`, `&T`, or `&mut T`, as
discussed before.

### Effect on ABI compatibility and semver

We have not yet proposed a comprehensive semver RFC (it's
coming). However, this RFC has some effect on what that RFC would say.
As discussed above, it is a breaking change for to add a blanket impl
for a `#[fundamental]` type. It is also a breaking change to add an
impl of a `#[fundamental]` trait to an existing type.

# Drawbacks

The primary drawback is that downstream crates cannot write an impl
over types other than references, such as `Option<LocalType>`. This
can be overcome by defining wrapper structs (new types), but that can
be annoying.

# Alternatives

- **Status quo.** In the status quo, the balance of power is heavily
  tilted towards child crates. Parent crates basically cannot add any
  impl for an existing trait to an existing type without potentially
  breaking child crates.

- **Take a hard line.** We could forego the `#[fundamental]` attribute, but
  it would force people to forego `Box<Trait>` impls as well as the
  useful closure-overloading pattern. This seems
  unfortunate. Moreover, it seems likely we will encounter further
  examples of "reasonable cases" that `#[fundamental]` can easily
  accommodate.

- **Specializations, negative impls, and contracts.** The gist
  referenced earlier includes [a section][c] covering various
  alternatives that I explored which came up short. These include
  specialization, explicit negative impls, and explicit contracts
  between the trait definer and the trait consumer.

# Unresolved questions

None.

[c]: https://gist.github.com/nikomatsakis/bbe6821b9e79dd3eb477#file-c-md
