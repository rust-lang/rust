- Feature Name: conservative_impl_trait
- Start Date: 2016-01-31
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Add a conservative form of abstract return types, aka `impl Trait`,
that will be compatible with most possible future extensions by
initially being restricted to:

- Only free-standing or inherent functions.
- Only return type position of a function.

Abstract return types allow a function to hide a concrete return
type behind a trait interface similar to trait objects, while
still generating the same statically dispatched code as with concrete types:

```rust
fn foo(n: u32) -> impl Iterator<Item=u32> {
    (0..n).map(|x| x * 100)
}
// ^ behaves as if it had return type Map<Range<u32>, Clos>
// where Clos = type of the |x| x * 100 closure.

for x in foo(10) {
    // ...
}

```

# Motivation
[motivation]: #motivation

> Why are we doing this? What use cases does it support? What is the expected outcome?

There has been much discussion around the `impl Trait` feature already, with
different proposals extending the core idea into different directions.

See http://aturon.github.io/blog/2015/09/28/impl-trait/ for detailed motivation, and
https://github.com/rust-lang/rfcs/pull/105 and https://github.com/rust-lang/rfcs/pull/1305 for prior RFCs on this topic.

It is not yet clear which, if any, of the proposals will end up as the "final form"
of the feature, so this RFC aims to only specify a usable subset that will
be compatible with most of them.

# Detailed design
[design]: #detailed-design

> This is the bulk of the RFC. Explain the design in enough detail for somebody familiar
> with the language to understand, and for somebody familiar with the compiler to implement.
> This should get into specifics and corner-cases, and include examples of how the feature is used.

#### Syntax

Let's start with the bikeshed: The proposed syntax is `@Trait` in return type
position, composing like trait objects to forms like `@(Foo+Send+'a)`.

The reason for choosing a sigil is ergonomics: Whatever the exact final
implementation will be capable of, you'd want it to be as easy to read/write
as trait objects, or else the more performant and idiomatic option would
be the more verbose one, and thus probably less used.

The argument can be made this decreases the google-ability of Rust syntax
(and this doesn't even talk about the _old_ `@T` pointer semantic the internet is still littered with),
but this would be somewhat mitigated by the feature being supposedly used commonly once it lands,
and can be explained in the docs as being short for `abstract` or `anonym`.

If there are good reasons against `@`, there is also the choice of `~`.
All points from above still apply, except `~` is a bit rarer in language
syntaxes in general, and depending on keyboard layout somewhat harder to reach.

Finally, if there is a huge incentive _against_ new (old?) sigils in the language,
there is also the option of using keyword-based syntax like `impl Trait` or
`abstract Trait`, but this would add a verbosity overhead for a feature
that will be used somewhat commonly.

#### Semantic

The core semantic of the feature is described below. Note that the sections after
this one go into more detail on some of the design decisions.

- `@Trait` may only be written at return type position
  of a freestanding or inherent-impl function, not in trait definitions,
  closure traits, function pointers, or any non-return type position.
- The function body can return values of any type that implements Trait,
  but all return values need to be of the same type.
- Outside of the function body, the return type is only known to implement Trait.
- As an exception to the above, OIBITS like `Send` and `Sync` leak through an abstract return type.
- The return type is unnameable.
- The return type has a identity based on all generic parameters the
  function body is parametrized by, and by the location of the function
  in the module system. This means type equality behaves like this:
  ```rust
  fn foo<T: Trait>(t: T) -> @Trait {
    t
  }

  fn bar() -> @Trait {
    123
  }

  fn equal_type<T>(a: T, b: T) {}

  equal_type(bar(), bar())                      // OK
  equal_type(foo::<i32>(0), foo::<i32>(0))      // OK
  equal_type(bar(), foo::<i32>(0))              // ERROR, `@Trait {bar}` is not the same type as `@Trait {foo<i32>}`
  equal_type(foo::<bool>(false), foo::<i32>(0)) // ERROR, `@Trait {foo<bool>}` is not the same type as `@Trait {foo<i32>}`
  ```
- The function body can not see through its own return type, so code like this
  would be forbidden just like on the outside:
  ```rust
  fn sum_to(n: u32) -> @Display {
      if n == 0 {
        0
      } else {
        n + sum_to(n - 1)
      }
  }
  ```
- Abstract return types are considered `Sized`, just like all return types today.

#### Limitation to only retun type position

There have been various proposed additional places where abstract types
might be usable. For example, `fn x(y: @Trait)` as shorthand for
`fn x<T: Trait>(y: T)`.
Since the exact semantic and user experience for these
locations are yet unclear
(`@Trait` would effectively behave completely different before and after the `->`),
this has also been excluded from this proposal.

#### OIBIT semantic

OIBITs leak through an abstract return type. This might be considered controversial, since
it effectively opens a channel where the result of function-local type inference affects
item-level API, but has been deemed worth it for the following reasons:

- Ergonomics: Trait objects already have the issue of explicitly needing to
  declare `Send`/`Sync`-ability, and not extending this problem to abstract return types
  is desireable.
- Low real change, since the situation already exists with structs with private fields:
  - In both cases, a change to the private implementation might change whether a OIBIT is
    implemented or not.
  - In both cases, the existence of OIBIT impls is not visible without doc tools
  - In both cases, you can only assert the existence of OIBIT impls
    by adding explicit trait bounds either to the API or to the crate's testsuite.

This means, however, that it has to be considered a silent breaking change
to change a function with a abstract return type
in a way that removes OIBIT impls, which might be a problem.

#### Anonymity

A abstract return type can not be named - this is similar to how closures
and function items are already unnameable types, and might be considered
a problem because it makes it not possible to build explicitly typed API
around the return type of a function.

The current semantic has been chosen for consistency and simplicity,
since the issue already exists with closures and function items, and
a solution to them will also apply here.

For example, if named abstract types get added, then existing
abstract return types could get upgraded to having a name transparently.
Likewise, if `typeof` makes it into the language, then you could refer to the
return type of a function without naming it.

#### Type transparency in recursive functions

Functions with abstract return types can not see through their own return type,
making code like this not compile:

```rust
fn sum_to(n: u32) -> @Display {
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
fn sum_to(n: u32) -> @Display {
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

#### Not legal in function pointers/closure traits

Because `@Trait` defines a type tied to the concrete function body,
it does not make much sense to talk about it separately in a function signature,
so the syntax is forbidden there.

#### Compability with conditional trait bounds

On valid critique for the existing `@Trait` proposal is that it does not
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

Using just `-> @Iterator`, this would not be possible to reproduce.

Since there has been no proposals so far that would address this in a way
that would conflict with the fixed-trait-set case, this RFC punts on that issue as well.

#### Limitation to free/inherent functions

One important usecase of abstract retutn types is to use them in trait methods.

However, there is an issue with this, namely that in combinations with generic
trait methods, they are effectively equivalent to higher kinded types.
Which is an issue because Rust HKT story is not yet figured out, so
any "accidential implementation" might cause uninteded fallout.

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

As has been elaborated on above, there are various way this feature could be
extended and combined with the language, so implementing it might
cause issues down the road if limitations or incompatibilities become apparent.

# Alternatives
[alternatives]: #alternatives

> What other designs have been considered? What is the impact of not doing this?

See the links in the motivation section for a more detailed analysis.

But basically, with this feature certain things remain hard or impossible to do
in Rust, like returning a efficiently usable type parametricised by
types private to a function body, like a iterator adapter containing a closure.

# Unresolved questions
[unresolved]: #unresolved-questions

> What parts of the design are still TBD?

None for the core feature proposed here, but many for possible extensions as elaborated on in detailed design.
