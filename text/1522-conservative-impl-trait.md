- Feature Name: conservative_impl_trait
- Start Date: 2016-01-31
- RFC PR: https://github.com/rust-lang/rfcs/pull/1522
- Rust Issue: https://github.com/rust-lang/rust/issues/34511

# Summary
[summary]: #summary

Add a conservative form of abstract return types, also known as `impl
Trait`, that will be compatible with most possible future extensions
by initially being restricted to:

- Only free-standing or inherent functions.
- Only return type position of a function.

Abstract return types allow a function to hide a concrete return
type behind a trait interface similar to trait objects, while
still generating the same statically dispatched code as with concrete types.

With the placeholder syntax used in discussions so far,
abstract return types would be used roughly like this:

```rust
fn foo(n: u32) -> impl Iterator<Item=u32> {
    (0..n).map(|x| x * 100)
}
// ^ behaves as if it had return type Map<Range<u32>, Closure>
// where Closure = type of the |x| x * 100 closure.

for x in foo(10) {
    // x = 0, 100, 200, ...
}

```

# Background

There has been much discussion around the `impl Trait` feature already, with
different proposals extending the core idea into different directions:

- The [original proposal](https://github.com/rust-lang/rfcs/pull/105).
- A [blog post](http://aturon.github.io/blog/2015/09/28/impl-trait/) reviving
  the proposal and further exploring the design space.
- A [more recent proposal](https://github.com/rust-lang/rfcs/pull/1305) with a
  substantially more ambitious scope.

This RFC is an attempt to make progress on the feature by proposing a minimal
subset that should be forwards-compatible with a whole range of extensions that
have been discussed (and will be reviewed in this RFC). However, even this small
step requires resolving some of the core questions raised in
[the blog post](http://aturon.github.io/blog/2015/09/28/impl-trait/).

This RFC is closest in spirit to the
[original RFC](https://github.com/rust-lang/rfcs/pull/105), and we'll repeat
its motivation and some other parts of its text below.

# Motivation
[motivation]: #motivation

> Why are we doing this? What use cases does it support? What is the expected outcome?

In today's Rust, you can write a function signature like

````rust
fn consume_iter_static<I: Iterator<u8>>(iter: I)
fn consume_iter_dynamic(iter: Box<Iterator<u8>>)
````

In both cases, the function does not depend on the exact type of the argument.
The type is held "abstract", and is assumed only to satisfy a trait bound.

* In the `_static` version using generics, each use of the function is
 specialized to a concrete, statically-known type, giving static dispatch, inline
 layout, and other performance wins.

* In the `_dynamic` version using trait objects, the concrete argument type is
  only known at runtime using a vtable.

On the other hand, while you can write

````rust
fn produce_iter_dynamic() -> Box<Iterator<u8>>
````

you _cannot_ write something like

````rust
fn produce_iter_static() -> Iterator<u8>
````

That is, in today's Rust, abstract return types can only be written using trait
objects, which can be a significant performance penalty. This RFC proposes
"unboxed abstract types" as a way of achieving signatures like
`produce_iter_static`. Like generics, unboxed abstract types guarantee static
dispatch and inline data layout.

Here are some problems that unboxed abstract types solve or mitigate:

* _Returning unboxed closures_. Closure syntax generates an anonymous type
  implementing a closure trait. Without unboxed abstract types, there is no way
  to use this syntax while returning the resulting closure unboxed, because there
  is no way to write the name of the generated type.

* _Leaky APIs_. Functions can easily leak implementation details in their return
  type, when the API should really only promise a trait bound. For example, a
  function returning `Rev<Splits<'a, u8>>` is revealing exactly how the iterator
  is constructed, when the function should only promise that it returns _some_
  type implementing `Iterator<u8>`. Using newtypes/structs with private fields
  helps, but is extra work. Unboxed abstract types make it as easy to promise only
  a trait bound as it is to return a concrete type.

* _Complex types_. Use of iterators in particular can lead to huge types:

  ````rust
  Chain<Map<'a, (int, u8), u16, Enumerate<Filter<'a, u8, vec::MoveItems<u8>>>>, SkipWhile<'a, u16, Map<'a, &u16, u16, slice::Items<u16>>>>
  ````

  Even when using newtypes to hide the details, the type still has to be written
  out, which can be very painful. Unboxed abstract types only require writing the
  trait bound.

* _Documentation_. In today's Rust, reading the documentation for the `Iterator`
  trait is needlessly difficult. Many of the methods return new iterators, but
  currently each one returns a different type (`Chain`, `Zip`, `Map`, `Filter`,
  etc), and it requires drilling down into each of these types to determine what
  kind of iterator they produce.

In short, unboxed abstract types make it easy for a function signature to
promise nothing more than a trait bound, and do not generally require the
function's author to write down the concrete type implementing the bound.

# Detailed design
[design]: #detailed-design

As explained at the start of the RFC, the focus here is a relatively narrow
introduction of abstract types limited to the return type of inherent methods
and free functions. While we still need to resolve some of the core questions
about what an "abstract type" means even in these cases, we avoid some of the
complexities that come along with allowing the feature in other locations or
with other extensions.

## Syntax

Let's start with the bikeshed: The proposed syntax is `impl Trait` in return type
position, composing like trait objects to forms like `impl Foo+Send+'a`.

It can be explained as "a type that implements `Trait`",
and has been used in that form in most earlier discussions and proposals.

Initial versions of this RFC proposed `@Trait` for brevity reasons,
since the feature is supposed to be used commonly once implemented,
but due to strong negative reactions by the community this has been
changed back to the current form.

There are other possibilities, like `abstract Trait` or `~Trait`, with
good reasons for or against them, but since the concrete choice of syntax
is not a blocker for the implementation of this RFC, it is intended for
a possible follow-up RFC to address syntax changes if needed.

## Semantics

The core semantics of the feature is described below.

Note that the sections after this one go into more detail on some of the design
decisions, and that **it is likely for many of the mentioned limitations to be
lifted at some point in the future**. For clarity, we'll separately categories the *core
semantics* of the feature (aspects that would stay unchanged with future extensions)
and the *initial limitations* (which are likely to be lifted later).

**Core semantics**:

- If a function returns `impl Trait`, its body can return values of any type that
  implements `Trait`, but all return values need to be of the same type.

- As far as the typesystem and the compiler is concerned, the return type
  outside of the function would not be a entirely "new" type, nor would it be a
  simple type alias. Rather, its semantics would be very similar to that of
  _generic type parameters_ inside a function, with small differences caused by
  being an _output_ rather than an _input_ of the function.

  - The type would be known to implement the specified traits.
  - The type would not be known to implement any other trait, with
    the exception of OIBITS (aka "auto traits") and default traits like `Sized`.
  - The type would not be considered equal to the actual underlying type.
  - The type would not be allowed to appear as the Self type for an `impl` block.

- Because OIBITS like `Send` and `Sync` will leak through an abstract return
  type, there will be some additional complexity in the compiler due to some
  non-local type checking becoming necessary.

- The return type has an identity based on all generic parameters the
  function body is parameterized by, and by the location of the function
  in the module system. This means type equality behaves like this:

  ```rust
  fn foo<T: Trait>(t: T) -> impl Trait {
      t
  }

  fn bar() -> impl Trait {
      123
  }

  fn equal_type<T>(a: T, b: T) {}

  equal_type(bar(), bar());                      // OK
  equal_type(foo::<i32>(0), foo::<i32>(0));      // OK
  equal_type(bar(), foo::<i32>(0));              // ERROR, `impl Trait {bar}` is not the same type as `impl Trait {foo<i32>}`
  equal_type(foo::<bool>(false), foo::<i32>(0)); // ERROR, `impl Trait {foo<bool>}` is not the same type as `impl Trait {foo<i32>}`
  ```

- The code generation passes of the compiler would not draw a distinction
  between the abstract return type and the underlying type, just like they don't
  for generic parameters. This means:
  - The same trait code would be instantiated, for example, `-> impl Any`
    would return the type id of the underlying type.
    - Specialization would specialize based on the underlying type.

**Initial limitations**:

- `impl Trait` may only be written within the return type of a freestanding or
  inherent-impl function, not in trait definitions or any non-return type position. They may also not appear
  in the return type of closure traits or function pointers,
  unless these are themselves part of a legal return type.

  - Eventually, we will want to allow the feature to be used within traits, and
    like in argument position as well (as an ergonomic improvement over today's generics).
  - Using `impl Trait` multiple times in the same return type would be valid,
    like for example in `-> (impl Foo, impl Bar)`.

- The type produced when a function returns `impl Trait` would be effectively
  unnameable, just like closures and function items.

  - We will almost certainly want to lift this limitation in the long run, so
    that abstract return types can be placed into structs and so on. There are a
    few ways we could do so, all related to getting at the "output type" of a
    function given all of its generic arguments.

- The function body cannot see through its own return type, so code like this
  would be forbidden just like on the outside:

  ```rust
  fn sum_to(n: u32) -> impl Display {
      if n == 0 {
          0
      } else {
          n + sum_to(n - 1)
      }
  }
  ```

  - It's unclear whether we'll want to lift this limitation, but it should be possible to do so.

## Rationale

### Why this semantics for the return type?

There has been a lot of discussion about what the semantics of the return type
should be, with the theoretical extremes being "full return type inference" and
"fully abstract type that behaves like a autogenerated newtype wrapper". (This
was in fact the main focus of the
[blog post](http://aturon.github.io/blog/2015/09/28/impl-trait/) on `impl
Trait`.)

The design as chosen in this RFC lies somewhat in between those two, since it
allows OIBITs to leak through, and allows specialization to "see" the full type
being returned. That is, `impl Trait` does not attempt to be a "tightly sealed"
abstraction boundary. The rationale for this design is a mixture of pragmatics
and principles.

#### Specialization transparency

**Principles for specialization transparency**:

The [specialization RFC](https://github.com/rust-lang/rfcs/pull/1210) has given
us a basic principle for how to understand bounds in function generics: they
represent a *minimum* contract between the caller and the callee, in that the
caller must meet at least those bounds, and the callee must be prepared to work
with any type that meets at least those bounds. However, with specialization,
the callee may choose different behavior when additional bounds hold.

This RFC abides by a similar interpretation for return types: the signature
represents the minimum bound that the callee must satisfy, and the caller must
be prepared to work with any type that meets at least that bound. Again, with
specialization, the caller may dispatch on additional type information beyond
those bounds.

In other words, to the extent that returning `impl Trait` is intended to be
symmetric with taking a generic `T: Trait`, transparency with respect to
specialization maintains that symmetry.

**Pragmatics for specialization transparency**:

The practical reason we want `impl Trait` to be transparent to specialization is the
same as the reason we want specialization in the first place: to be able to
break through abstractions with more efficient special-case code.

This is particularly important for one of the primary intended usecases:
returning `impl Iterator`. We are very likely to employ specialization for various
iterator types, and making the underlying return type invisible to
specialization would lose out on those efficiency wins.

#### OIBIT transparency

OIBITs leak through an abstract return type. This might be considered controversial, since
it effectively opens a channel where the result of function-local type inference affects
item-level API, but has been deemed worth it for the following reasons:

- Ergonomics: Trait objects already have the issue of explicitly needing to
  declare `Send`/`Sync`-ability, and not extending this problem to abstract
  return types is desirable. In practice, most uses of this feature would have
  to add explicit bounds for OIBITS if they wanted to be maximally usable.

- Low real change, since the situation already somewhat exists on structs with private fields:
  - In both cases, a change to the private implementation might change whether a OIBIT is
    implemented or not.
  - In both cases, the existence of OIBIT impls is not visible without documentation tools
  - In both cases, you can only assert the existence of OIBIT impls
  by adding explicit trait bounds either to the API or to the crate's test suite.

In fact, a large part of the point of OIBITs in the first place was to cut
across abstraction barriers and provide information about a type without the
type's author having to explicitly opt in.

This means, however, that it has to be considered a silent breaking change to
change a function with a abstract return type in a way that removes OIBIT impls,
which might be a problem. (As noted above, this is already the case for `struct`
definitions.)

But since the number of used OIBITs is relatively small, deducing the return type
in a function body and reasoning about whether such a breakage will occur has
been deemed as a manageable amount of work.

#### Wherefore type abstraction?

In the [most recent RFC](https://github.com/rust-lang/rfcs/pull/1305) related to
this feature, a more "tightly sealed" abstraction mechanism was
proposed. However, part of the discussion on specialization centered on
precisely the issue of what type abstraction provides and how to achieve it.  A
particular salient point there is that, in Rust, *privacy* is already our
primary mechanism for hiding
(["privacy is the new parametricity"](https://github.com/rust-lang/rfcs/pull/1210#issuecomment-181992044)). In
practice, that means that if you want opacity against specialization, you should
use something like a newtype.

### Anonymity

A abstract return type cannot be named in this proposal, which means that it
cannot be placed into `structs` and so on. This is not a fundamental limitation
in any sense; the limitation is there both to keep this RFC simple, and because
the precise way we might want to allow naming of such types is still a bit
unclear. Some possibilities include a `typeof` operator, or explicit named
abstract types.

### Limitation to only return type position

There have been various proposed additional places where abstract types
might be usable. For example, `fn x(y: impl Trait)` as shorthand for
`fn x<T: Trait>(y: T)`.

Since the exact semantics and user experience for these locations are yet
unclear (`impl Trait` would effectively behave completely different before and after
the `->`), this has also been excluded from this proposal.

### Type transparency in recursive functions

Functions with abstract return types can not see through their own return type,
making code like this not compile:

```rust
fn sum_to(n: u32) -> impl Display {
    if n == 0 {
        0
    } else {
        n + sum_to(n - 1)
    }
}
```

This limitation exists because it is not clear how much a function body
can and should know about different instantiations of itself.

It would be safe to allow recursive calls if the set of generic parameters
is identical, and it might even be safe if the generic parameters are different,
since you would still be inside the private body of the function, just
differently instantiated.

But variance caused by lifetime parameters and the interaction with
specialization makes it uncertain whether this would be sound.

In any case, it can be initially worked around by defining a local helper function like this:

```rust
fn sum_to(n: u32) -> impl Display {
    fn sum_to_(n: u32) -> u32 {
        if n == 0 {
            0
        } else {
            n + sum_to_(n - 1)
        }
    }
    sum_to_(n)
}
```

### Not legal in function pointers/closure traits

Because `impl Trait` defines a type tied to the concrete function body,
it does not make much sense to talk about it separately in a function signature,
so the syntax is forbidden there.

### Compatibility with conditional trait bounds

On valid critique for the existing `impl Trait` proposal is that it does not
cover more complex scenarios, where the return type would implement
one or more traits depending on whether a type parameter does so with another.

For example, a iterator adapter might want to implement `Iterator` and
`DoubleEndedIterator`, depending on whether the adapted one does:

```rust
fn skip_one<I>(i: I) -> SkipOne<I> { ... }
struct SkipOne<I> { ... }
impl<I: Iterator> Iterator for SkipOne<I> { ... }
impl<I: DoubleEndedIterator> DoubleEndedIterator for SkipOne<I> { ... }
```

Using just `-> impl Iterator`, this would not be possible to reproduce.

Since there has been no proposals so far that would address this in a way
that would conflict with the fixed-trait-set case, this RFC punts on that issue as well.

### Limitation to free/inherent functions

One important usecase of abstract return types is to use them in trait methods.

However, there is an issue with this, namely that in combinations with generic
trait methods, they are effectively equivalent to higher kinded types.
Which is an issue because Rust HKT story is not yet figured out, so
any "accidental implementation" might cause unintended fallout.

HKT allows you to be generic over a type constructor, aka a
"thing with type parameters", and then instantiate them at some later point to
get the actual type.
For example, given a HK type `T` that takes one type as parameter, you could
write code that uses `T<u32>` or `T<bool>` without caring about
whether `T = Vec`, `T = Box`, etc.

Now if we look at abstract return types, we have a similar situation:

```rust
trait Foo {
    fn bar<U>() -> impl Baz
}
```

Given a `T: Foo`, we could instantiate `T::bar::<u32>` or `T::bar::<bool>`,
and could get arbitrary different return types of `bar` instantiated
with a `u32` or `bool`,
just like `T<u32>` and `T<bool>` might give us `Vec<u32>` or `Box<bool>`
in the example above.

The problem does not exists with trait method return types today because
they are concrete:

```rust
trait Foo {
    fn bar<U>() -> X<U>
}
```

Given the above code, there is no way for `bar` to choose a return type `X`
that could fundamentally differ between instantiations of `Self`
while still being instantiable with an arbitrary `U`.

At most you could return a associated type, but then you'd loose the generics
from `bar`

```rust
trait Foo {
    type X;
    fn bar<U>() -> Self::X // No way to apply U
}
```

So, in conclusion, since Rusts HKT story is not yet fleshed out,
and the compatibility of the current compiler with it is unknown,
it is not yet possible to reach a concrete solution here.

In addition to that, there are also different proposals as to whether
a abstract return type is its own thing or sugar for a associated type,
how it interacts with other associated items and so on,
so forbidding them in traits seems like the best initial course of action.

# Drawbacks
[drawbacks]: #drawbacks

> Why should we *not* do this?

## Drawbacks due to the proposal's minimalism

As has been elaborated on above, there are various way this feature could be
extended and combined with the language, so implementing it might cause issues
down the road if limitations or incompatibilities become apparent. However,
variations of this RFC's proposal have been under discussion for quite a long
time at this point, and this proposal is carefully designed to be
future-compatible with them, while resolving the core issue around transparency.

A drawback of limiting the feature to return type position (and not arguments)
is that it creates a somewhat inconsistent mental model: it forces you to
understand the feature in a highly special-cased way, rather than as a general
way to talk about unknown-but-bounded types in function signatures. This could
be particularly bewildering to newcomers, who must choose between `T: Trait`,
`Box<Trait>`, and `impl Trait`, with the latter only usable in one place.

## Drawbacks due to partial transparency

The fact that specialization and OIBITs can "see through" `impl Trait` may be
surprising, to the extent that one wants to see `impl Trait` as an abstraction
mechanism. However, as the RFC argued in the rationale section, this design is
probably the most consistent with our existing post-specialization abstraction
mechanisms, and lead to the relatively simple story that *privacy* is the way to
achieve hiding in Rust.

# Alternatives
[alternatives]: #alternatives

> What other designs have been considered? What is the impact of not doing this?

See the links in the motivation section for detailed analysis that we won't
repeat here.

But basically, without this feature certain things remain hard or impossible to do
in Rust, like returning a efficiently usable type parameterized by
types private to a function body, for example an iterator adapter containing a closure.

# Unresolved questions
[unresolved]: #unresolved-questions

> What parts of the design are still to be determined?

The precise implementation details for OIBIT transparency are a bit unclear: in
general, it means that type checking may need to proceed in a particular order,
since you cannot get the full type information from the signature alone (you
have to typecheck the function body to determine which OIBITs apply).
