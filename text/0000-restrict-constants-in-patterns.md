- Feature Name: (fill me in with a unique ident, my_awesome_feature)
- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Feature-gate the use of constants in patterns unless those constants
have simple types, like integers, booleans, and characters. The
semantics of constants in general were never widely discussed and the
compiler's current implementation is not broadly agreed upon (though
it has many proponents). The intention of adding a feature-gate is to
give us time to discuss and settle on the desired semantics in an
"affirmative" way.

Because the compiler currently accepts a larger set of constants, this
is a backwards incompatible change. This is justified as part of the
["underspecified language semantics" clause of RFC 1122][ls]. A
[crater run] found 14 regressions on crates.io, which suggests that
the impact of this change on real code would be minimal.

Note: this was also discussed on an [internals thread]. Major points
from that thread are summarized either inline or in alternatives.

[ls]: https://github.com/rust-lang/rfcs/blob/master/text/1122-language-semver.md#underspecified-language-semantics
[crater run]: https://gist.github.com/nikomatsakis/26096ec2a2df3c1fb224
[internals thread]: https://internals.rust-lang.org/t/how-to-handle-pattern-matching-on-constants/2846)

# Motivation
[motivation]: #motivation

The compiler currently permits any kind of constant to be used within
a pattern. However, the *meaning* of such a pattern is somewhat
controversial: the current semantics implemented by the compiler were
[adopted in July of 2014](https://github.com/rust-lang/rust/pull/15650)
and were never widely discussed nor did they go through the RFC
process. Moreover, the discussion at the time was focused primarily on
implementation concerns, and overlooked the potential semantic
hazards.

### Semantic vs structural equality

Consider a program like this one, which references a constant value
from within a pattern:

```rust
struct SomeType {
    a: u32,
    b: u32,
}

const SOME_CONSTANT: SomeType = SomeType { a: 22+22, b: 44+44 };

fn test(v: SomeType) {
    match v {
        SOME_CONSTANT => println!("Yes"),
        _ => println!("No"),
    }
}
```

The question at hand is what do we expect this match to do, precisely?
There are two main possibilities: semantic and structural equality.

**Semantic equality.** Semantic equality states that a pattern
`SOME_CONSTANT` matches a value `v` if `v == SOME_CONSTANT`. In other
words, the `match` statement above would be exactly equivalent to an
`if`:

```rust
if v == SOME_CONSTANT {
    println!("Yes")
} else {
    println!("No");
}
```

Under semantic equality, the program above would not compile, because
`SomeType` does not implement the `PartialEq` trait.

**Structural equality.** Under structural equality, `v` matches the
pattern `SOME_CONSTANT` if all of its fields are (structurally) equal.
Primitive types like `u32` are structurally equal if they represent
the same value (but see below for discussion about floating point
types like `f32` and `f64`). This means that the `match` statement
above would be roughly equivalent to the following `if` (modulo
privacy):

```rust
if v.a == SOME_CONSTANT.a && v.b == SOME_CONSTANT.b {
    println!("Yes")
} else {
    println!("No");
}
```

Structural equality basically says "two things are structurally equal
if their fields are structurally equal". It is sort of equality you
would get if everyone used `#[derive(PartialEq)]` on all types. Note
that the equality defined by structural equality is completely
distinct from the `==` operator, which is tied to the `PartialEq`
traits. That is, two values that are *semantically unequal* could be
*structurally equal* (an example where this might occur is the
floating point value `NaN`).

**Current semantics.** The compiler's current semantics are basically
structural equality, though in the case of floating point numbers they
are arguably closer to semantic equality (details below). In
particular, when a constant appears in a pattern, the compiler first
evaluates that constant to a specific value. So we would reduce the
expression:

```rust
const SOME_CONSTANT: SomeType = SomeType { a: 22+22, b: 44+44 };
```

to the value `SomeType { a: 44, b: 88 }`. We then expand the pattern
`SOME_CONSTANT` as though you had typed this value in place (well,
almost as though, read on for some complications around privacy).
Thus the match statement above is equivalent to:

```rust
match v {
    SomeType { a: 44, b: 88 } => println!(Yes),
    _ => println!("No"),
}
```

### Disadvantages of the current approach

Given that the compiler already has a defined semantics, it is
reasonable to ask why we might want to change it. There
are two main disadvantages:

1. **No abstraction boundary.** The current approach does not permit
   types to define what equality means for themselves (at least not if
   they can be constructed in a constant).
2. **Scaling to associated constants.** The current approach does not
   permit associated constants or generic integers to be used in a
   match statement.
   
#### Disadvantage: Weakened abstraction bounary

The single biggest concern with structural equality is that it
introduces two distinct notions of equality: the `==` operator, based
on the `PartialEq` trait, and pattern matching, based on a builtin
structural recursion. This will cause problems for user-defined types
that rely on `PartialEq` to define equality. Put another way, **it is
no longer possible for user-defined types to completely define what
equality means for themselves** (at least not if they can be
constructed in a constant). Furthermore, because the builtin
structural recursion does not consider privacy, `match` statements can
now be used to **observe private fields**.

**Example: Normalized durations.** Consider a simple duration type:

```rust
#[derive(Copy, Clone)]
pub struct Duration {
    pub seconds: u32,
    pub minutes: u32,
}  
```

Let's say that this `Duration` type wishes to represent a span of
time, but it also wishes to preserve whether that time was expressed
in seconds or minutes.  In other words, 60 seconds and 1 minute are
equal values, but we don't want to normalize 60 seconds into 1 minute;
perhaps because it comes from user input and we wish to keep things
just as the user chose to express it.

We might implement `PartialEq` like so (actually the `PartialEq` trait
is slightly different, but you get the idea):

```rust
impl PartialEq for Duration {
    fn eq(&self, other: &Duration) -> bool {
        let s1 = (self.seconds as u64) + (self.minutes as u64 * 60);
        let s2 = (other.seconds as u64) + (other.minutes as u64 * 60);
        s1 == s2
    }
}
```

Now imagine I have some constants:

```rust
const TWENTY_TWO_SECONDS: Duration = Duration { seconds: 22, minutes: 0 };
const ONE_MINUTE: Duration = Duration { seconds: 0, minutes: 1 };
```

And I write a match statement using those constants:

```rust
fn detect_some_case_or_other(d: Duration) {
    match d {
        TWENTY_TWO_SECONDS => /* do something */,
        ONE_MINUTE => /* do something else */,
        _ => /* do something else again */,
    }
}
```

Now this code is, in all probability, buggy. Probably I meant to use
the notion of equality that `Duration` defined, where seconds and
minutes are normalized. But that is not the behavior I will see --
instead I will use a pure structural match. What's worse, this means
the code will probably work in my local tests, since I like to say
"one minute", but it will break when I demo it for my customer, since
she prefers to write "60 seconds".

**Example: Floating point numbers.** Another example is floating point
numbers. Consider the case of `0.0` and `-0.0`: these two values are
distinct, but they typically behave the same; so much so that they
compare equal (that is, `0.0 == -0.0` is `true`).  So it is likely
that code such as:

```rust
match some_computation() {
    0.0 => ...,
    x => ...,
}
```

did not intend to discriminate between zero and negative zero.  In
fact, in the compiler today, match *will* compare 0.0 and -0.0 as
equal. We simply do not extend that courtesy to user-defined types.

**Example: observing private fields.** The current constant expansion
code does not consider privacy. In other words, constants are expanded
into equivalent patterns, but those patterns may not have been
something the user could have typed because of privacy rules. Consider
a module like:

```rust
mod foo {
    pub struct Foo { b: bool }
    pub const V1: Foo = Foo { b: true };
    pub const V2: Foo = Foo { b: false };
}
```

Note that there is an abstraction boundary here: b is a private
field. But now if I wrote code from another module that matches on a
value of type Foo, that abstraction boundary is pierced:

```rust
fn bar(f: x::Foo) {
    // rustc knows this is exhaustive because if expanded `V1` into
    // equivalent patterns; patterns you could not write by hand!
    match f {
        x::V1 => { /* moreover, now we know that f.b is true */ }
        x::V2 => { /* and here we know it is false */ }
    }
}
```

Note that, because `Foo` does not implement `PartialEq`, just having
access to `V1` would not otherwise allow us to observe the value of
`f.b`. (And even if `Foo` *did* implement `PartialEq`, that
implementation might not read `f.b`, so we still would not be able to
observe its value.)

**More examples.** There are numerous possible examples here. For
example, strings that compare using case-insensitive comparisons, but
retain the original case for reference, such as those used in
file-systems. Views that extract a subportion of a larger value (and
hence which should only compare that subportion). And so forth.

#### Disadvantage: Scaling to associated constants and generic integers

Rewriting constants into patterns requires that we can **fully
evaluate** the constant at the time of exhaustiveness checking. For
associated constants and type-level integers, that is not possible --
we have to wait until monomorphization time. Consider:

```rust
trait SomeTrait {
    const A: bool;
    const B: bool;
}

fn foo<T:SomeTrait>(x: bool) {
    match x {
        T::A => println!("A"),
        T::B => println!("B"),
    }
}

impl SomeTrait for i32 {
    const A: bool = true;
    const B: bool = true;
}    

impl SomeTrait for u32 {
    const A: bool = true;
    const B: bool = false;
}    
```

Is this match exhaustive? Does it contain dead code? The answer will
depend on whether `T=i32` or `T=u32`, of course.

### Advantages of the current approach

However, structural equality also has a number of advantages:

**Better optimization.** One of the biggest "pros" is that it can
potentially enable nice optimization. For example, given constants like the following:

```rust
struct Value { x: u32 }
const V1: Value = Value { x: 0 };
const V2: Value = Value { x: 1 };
const V3: Value = Value { x: 2 };
const V4: Value = Value { x: 3 };
const V5: Value = Value { x: 4 };
```

and a match pattern like the following:

```rust
match v {
    V1 => ..., 
    ...,
    V5 => ...,
}
```

then, because pattern matching is always a process of structurally
extracting values, we can compile this to code that reads the field
`x` (which is a `u32`) and does an appropriate switch on that
value. Semantic equality would potentially force a more conservative
compilation strategy.

**Better exhautiveness and dead-code checking.** Similarly, we can do
more thorough exhaustiveness and dead-code checking. So for example if
I have a struct like:

```rust
struct Value { field: bool }
const TRUE: Value { field: true };
const FALSE: Value { field: false };
```

and a match pattern like:

```rust
match v { TRUE => .., FALSE => .. }
```

then we can prove that this match is exhaustive. Similarly, we can prove
that the following match contains dead-code:

```rust
const A: Value { field: true };
match v {
    TRUE => ...,
    A => ...,
}
```

Again, some of the alternatives might not allow this. (But note the
cons, which also raise the question of exhaustiveness checking.)

**Nullary variants and constants are (more) equivalent.** Currently,
there is a sort of equivalence between enum variants and constants, at
least with respect to pattern matching. Consider a C-like enum:

```rust
enum Modes {
    Happy = 22,
    Shiny = 44,
    People = 66,
    Holding = 88,
    Hands = 110,
}

const C: Modes = Modes::Happy;
```

Now if I match against `Modes::Happy`, that is matching against an
enum variant, and under *all* the proposals I will discuss below, it
will check the actual variant of the value being matched (regardless
of whether `Modes` implements `PartialEq`, which it does not here). On
the other hand, if matching against `C` were to require a `PartialEq`
impl, then it would be illegal. Therefore matching against an *enum
variant* is distinct from matching against a *constant*.

# Detailed design
[design]: #detailed-design

Define the set of builtin types `B` as follows:

```
B = i8 | i16 | i32 | i64 | isize  // signed integers
  | u8 | u16 | u32 | u64 | usize  // unsigned integers
  | char                          // characters 
  | bool                          // booleans
  | (B, ..., B)                   // tuples of builtin types
```

Any constants appearing in a pattern whose type is not a member of `B`
will be feature-gated. This feature-gate will be phased in using a
deprecation cycle, as usual.

# Drawbacks
[drawbacks]: #drawbacks

This is a breaking change, which means some people will have to change
their code. Moreover, code that is currently using constants of disallowed
types becomes slightly more verbose. For example:

```rust
match foo {
    Some(CONSTANT) => ...,
    None => ...,
}
```

would now be written:

```rust
match foo {
    Some(v) if v == CONSTANT => ...,
    None => ...,
}
```

# Alternatives
[alternatives]: #alternatives

**No changes.** Naturally we could opt to keep the semantics as they
are. The advantages and disadvantages are discussed above.

**Embrace semantic equality.** We could opt to just go straight
towards "semantic equality". Howver, it seems better to reset the
semantics to a base point that everyone can agree on, and then extend
from that base point. Moreover, adopting semantic equality straight
out would be a riskier breaking change, as it could silently change
the semantics of existing programs (whereas the current proposal only
causes compilation to fail, never changes what an existing program
will do).

# Discussion thread summary

This section summarizes various points that were raised in the
[internals thread] which are related to patterns but didn't seem to
fit elsewhere.

**Overloaded patterns.** Some languages, notably Scala, permit
overloading of patterns. This is related to "semantic equality" in
that it involves executing custom, user-provided code at compilation
time.

**Pattern synonyms.** Haskell offers a feature called "pattern
synonyms" and
[it was argued](https://internals.rust-lang.org/t/how-to-handle-pattern-matching-on-constants/2846/39?u=nikomatsakis)
that the current treatment of patterns can be viewed as a similar
feature. This may be true, but constants-in-patterns are lacking a
number of important features from pattern synonyms, such as bindings,
as
[discussed in this response](https://internals.rust-lang.org/t/how-to-handle-pattern-matching-on-constants/2846/48?u=nikomatsakis).
The author feels that pattern synonyms might be a useful feature, but
it would be better to design them as a first-class feature, not adapt
constants for that purpose.

# Unresolved questions
[unresolved]: #unresolved-questions

**Should we also adjust the exhaustiveness and match analysis
algorithm to be more conservative around constants?** This RFC just
proposes limiting the types of constants that can be used in a match
pattern. However, since the code currently inlines the actual values
of constants before doing exhaustiveness checking, this also implies
that it can compute exhaustiveness and dead-code in cases where it
arguably should not be able to.

For example, the following code
[fails to compile](http://is.gd/PJjNKl) because it contains dead-code:

```rust
const X: u64 = 0;
const Y: u64 = 0;
fn bar(foo: u64) {
    match foo {
        X => { }
        Y => { }
        _ => { }
    }
}
```

However, we would be unable to perform such an analysis in a more
generic context, such as with an associated constant:

```rust
trait Trait {
    const X: u64;
    const Y: u64;
}

fn bar<T:Trait>(foo: u64) {
    match foo {
        T::X => { }
        T::Y => { }
        _ => { }
    }
}
```

Here, although it may well be that `T::X == T::Y`, we can't know for
sure. So, for consistency, we may wish to treat all constants opaquely
regardless of whether we are in a generic context or not.

Another argument in favor of treating all constants opaquely is that
the current behavior can leak details that perhaps were intended to be
hidden. For example, imagine that I define a fn `hash` that, given a
previous hash and a value, produces a new hash.  Because I am lazy and
prototyping my system, I decide for now to just ignore the new value
and pass the old hash through:

```rust
const fn add_to_hash(prev_hash: u64, _value: u64) -> u64 {
    prev_hash
}
```

Now I have some consumers of my library and they define a few constants:

```rust
const HASH_OF_ZERO: add_to_hash(0, 0);
const HASH_OF_ONE: add_to_hash(0, 1);
```

And at some point they write a match statement:

```rust
fn process_hash(h: u64) {
    match h {
        HASH_OF_ZERO => /* do something */,
        HASH_OF_ONE => /* do something else */,
        _ => /* do something else again */,
}
```

As before, what you get when you [compile this](http://is.gd/u5WtCo)
is a dead-code error, because the compiler can see that `HASH_OF_ZERO`
and `HASH_OF_ONE` are the same value.

Part of the solution here might be making "unreachable patterns" a
warning and not an error. The author feels this would be a good idea
regardless (though not necessarily as part of this RFC). However,
that's not a complete solution, since -- at least for `bool` constants
-- the same issues arise if you consider exhaustiveness checking.

On the other hand, it feels very silly for the compiler not to
understand that `match some_bool { true => ..., false => ... }` is
exhaustive. Furthermore, there are other ways for the values of
constants to "leak out", such as when part of a type like
`[u8; SOME_CONSTANT]` (a point made by both [arielb1][arielb1ac] and
[glaebhoerl][gac] on the [internals thread]). Therefore, the proper
way to address this question is perhaps to consider an explicit form
of "abstract constant".

[arielb1ac]: https://internals.rust-lang.org/t/how-to-handle-pattern-matching-on-constants/2846/9?u=nikomatsakis
[gac]: https://internals.rust-lang.org/t/how-to-handle-pattern-matching-on-constants/2846/32?u=nikomatsakis
