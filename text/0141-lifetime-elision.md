- Start Date: (2014-06-24)
- RFC PR: [rust-lang/rfcs#141](https://github.com/rust-lang/rfcs/pull/141)
- Rust Issue: [rust-lang/rust#15552](https://github.com/rust-lang/rust/issues/15552)

# Summary

This RFC proposes to

1. Expand the rules for eliding lifetimes in `fn` definitions, and
2. Follow the same rules in `impl` headers.

By doing so, we can avoid writing lifetime annotations ~87% of the time that
they are currently required, based on a survey of the standard library.

# Motivation

In today's Rust, lifetime annotations make code more verbose, both for methods

```rust
fn get_mut<'a>(&'a mut self) -> &'a mut T
```

and for `impl` blocks:

```rust
impl<'a> Reader for BufReader<'a> { ... }
```

In the vast majority of cases, however, the lifetimes follow a very simple
pattern.

By codifying this pattern into simple rules for filling in elided lifetimes, we
can avoid writing any lifetimes in ~87% of the cases where they are currently
required.

Doing so is a clear ergonomic win.

# Detailed design

## Today's lifetime elision rules

Rust currently supports eliding lifetimes in functions, so that you can write

```rust
fn print(s: &str);
fn get_str() -> &str;
```

instead of

```rust
fn print<'a>(s: &'a str);
fn get_str<'a>() -> &'a str;
```

The elision rules work well for functions that consume references, but not for
functions that produce them. The `get_str` signature above, for example,
promises to produce a string slice that lives arbitrarily long, and is
either incorrect or should be replaced by

```rust
fn get_str() -> &'static str;
```

Returning `'static` is relatively rare, and it has been proposed to make leaving
off the lifetime in output position an error for this reason.

Moreover, lifetimes cannot be elided in `impl` headers.

## The proposed rules

### Overview

This RFC proposes two changes to the lifetime elision rules:

1. Since eliding a lifetime in output position is usually wrong or undesirable
   under today's elision rules, interpret it in a different and more useful way.

2. Interpret elided lifetimes for `impl` headers analogously to `fn` definitions.

### Lifetime positions

A _lifetime position_ is anywhere you can write a lifetime in a type:

```rust
&'a T
&'a mut T
T<'a>
```

As with today's Rust, the proposed elision rules do _not_ distinguish between
different lifetime positions. For example, both `&str` and `Ref<uint>` have
elided a single lifetime.

Lifetime positions can appear as either "input" or "output":

* For `fn` definitions, input refers to the types of the formal arguments
  in the `fn` definition, while output refers to
  result types. So `fn foo(s: &str) -> (&str, &str)` has elided one lifetime in
  input position and two lifetimes in output position.
  Note that the input positions of a `fn` method definition do not
  include the lifetimes that occur in the method's `impl` header
  (nor lifetimes that occur in the trait header, for a default method).


* For `impl` headers, input refers to the lifetimes appears in the type
  receiving the `impl`, while output refers to the trait, if any. So `impl<'a>
  Foo<'a>` has `'a` in input position, while `impl<'a, 'b, 'c>
  SomeTrait<'b, 'c> for Foo<'a, 'c>` has `'a` in input position, `'b`
  in output position, and `'c` in both input and output positions.

### The rules

* Each elided lifetime in input position becomes a distinct lifetime
  parameter. This is the current behavior for `fn` definitions.

* If there is exactly one input lifetime position (elided or not), that lifetime
  is assigned to _all_ elided output lifetimes.

* If there are multiple input lifetime positions, but one of them is `&self` or
  `&mut self`, the lifetime of `self` is assigned to _all_ elided output lifetimes.

* Otherwise, it is an error to elide an output lifetime.

Notice that the _actual_ signature of a `fn` or `impl` is based on the expansion
rules above; the elided form is just a shorthand.

### Examples

```rust
fn print(s: &str);                                      // elided
fn print<'a>(s: &'a str);                               // expanded

fn debug(lvl: uint, s: &str);                           // elided
fn debug<'a>(lvl: uint, s: &'a str);                    // expanded

fn substr(s: &str, until: uint) -> &str;                // elided
fn substr<'a>(s: &'a str, until: uint) -> &'a str;      // expanded

fn get_str() -> &str;                                   // ILLEGAL

fn frob(s: &str, t: &str) -> &str;                      // ILLEGAL

fn get_mut(&mut self) -> &mut T;                        // elided
fn get_mut<'a>(&'a mut self) -> &'a mut T;              // expanded

fn args<T:ToCStr>(&mut self, args: &[T]) -> &mut Command                  // elided
fn args<'a, 'b, T:ToCStr>(&'a mut self, args: &'b [T]) -> &'a mut Command // expanded

fn new(buf: &mut [u8]) -> BufWriter;                    // elided
fn new<'a>(buf: &'a mut [u8]) -> BufWriter<'a>          // expanded

impl Reader for BufReader { ... }                       // elided
impl<'a> Reader for BufReader<'a> { .. }                // expanded

impl Reader for (&str, &str) { ... }                    // elided
impl<'a, 'b> Reader for (&'a str, &'b str) { ... }      // expanded

impl StrSlice for &str { ... }                          // elided
impl<'a> StrSlice<'a> for &'a str { ... }               // expanded

trait Bar<'a> { fn bound(&'a self) -> &int { ... }    fn fresh(&self) -> &int { ... } }           // elided
trait Bar<'a> { fn bound(&'a self) -> &'a int { ... } fn fresh<'b>(&'b self) -> &'b int { ... } } // expanded

impl<'a> Bar<'a> for &'a str {
  fn bound(&'a self) -> &'a int { ... } fn fresh(&self) -> &int { ... }              // elided
}
impl<'a> Bar<'a> for &'a str {
  fn bound(&'a self) -> &'a int { ... } fn fresh<'b>(&'b self) -> &'b int { ... }    // expanded
}

// Note that when the impl reuses the same signature (with the same elisions)
// from the trait definition, the expanded forms will also match, and thus
// the `impl` will be compatible with the `trait`.

impl Bar for &str            { fn bound(&self) -> &int { ... } }           // elided
impl<'a> Bar<'a> for &'a str { fn bound<'b>(&'b self) -> &'b int { ... } } // expanded

// Note that the preceding example's expanded methods do not match the
// signatures from the above trait definition for `Bar`; in the general
// case, if the elided signatures between the `impl` and the `trait` do
// not match, an expanded `impl` may not be compatible with the given
// `trait` (and thus would not compile).

impl Bar for &str            { fn fresh(&self) -> &int { ... } }           // elided
impl<'a> Bar<'a> for &'a str { fn fresh<'b>(&'b self) -> &'b int { ... } } // expanded

impl Bar for &str {
  fn bound(&'a self) -> &'a int { ... } fn fresh(&self) -> &int { ... }    // ILLEGAL: unbound 'a
}

```

## Error messages

Since the shorthand described above should eliminate most uses of explicit
lifetimes, there is a potential "cliff". When a programmer first encounters a
situation that requires explicit annotations, it is important that the compiler
gently guide them toward the concept of lifetimes.

An error can arise with the above shorthand only when the program elides an
output lifetime and neither of the rules can determine how to annotate it.

### For `fn`

The error message should guide the programmer toward the concept of lifetime by
talking about borrowed values:

> This function's return type contains a borrowed value, but the signature does
> not say which parameter it is borrowed from. It could be one of a, b, or
> c. Mark the input parameter it borrows from using lifetimes,
> e.g. [generated example]. See [url] for an introduction to lifetimes.

This message is slightly inaccurate, since the presence of a lifetime parameter
does not necessarily imply the presence of a borrowed value, but there are no
known use-cases of phantom lifetime parameters.

### For `impl`

The error case on `impl` is exceedingly rare: it requires (1) that the `impl` is
for a trait with a lifetime argument, which is uncommon, and (2) that the `Self`
type has multiple lifetime arguments.

Since there are no clear "borrowed values" for an `impl`, this error message
speaks directly in terms of lifetimes. This choice seems warranted given that a
programmer implementing a trait with lifetime parameters will almost certainly
already understand lifetimes.

> TraitName requires lifetime arguments, and the impl does not say which
> lifetime parameters of TypeName to use. Mark the parameters explicitly,
> e.g. [generated example]. See [url] for an introduction to lifetimes.

## The impact

To assess the value of the proposed rules, we conducted a survey of the code
defined _in_ `libstd` (as opposed to the code it reexports). This corpus is
large and central enough to be representative, but small enough to easily
analyze.

We found that of the 169 lifetimes that currently require annotation for
`libstd`, 147 would be elidable under the new rules, or 87%.

_Note: this percentage does not include the large number of lifetimes that are
already elided with today's rules._

The detailed data is available at:
https://gist.github.com/aturon/da49a6d00099fdb0e861

# Drawbacks

## Learning lifetimes

The main drawback of this change is pedagogical. If lifetime annotations are
rarely used, newcomers may encounter error messages about lifetimes long before
encountering lifetimes in signatures, which may be confusing. Counterpoints:

* This is already the case, to some extent, with the current elision rules.

* Most existing error messages are geared to talk about specific borrows not
  living long enough, pinpointing their _locations_ in the source, rather than
  talking in terms of lifetime annotations. When the errors do mention
  annotations, it is usually to suggest specific ones.

* The proposed error messages above will help programmers transition out of the
  fully elided regime when they first encounter a signature requiring it.

* When combined with a good tutorial on the borrow/lifetime system (which should
  be introduced early in the documentation), the above should provide a
  reasonably gentle path toward using and understanding explicit lifetimes.

Programmers learn lifetimes once, but will use them many times. Better to favor
long-term ergonomics, if a simple elision rule can cover 87% of current lifetime
uses (let alone the currently elided cases).

## Subtlety for non-`&` types

While the rules are quite simple and regular, they can be subtle when applied to
types with lifetime positions. To determine whether the signature

```rust
fn foo(r: Bar) -> Bar
```

is actually using lifetimes via the elision rules, you have to know whether
`Bar` has a lifetime parameter. But this subtlety already exists with the
current elision rules. The benefit is that library types like `Ref<'a, T>` get
the same status and ergonomics as built-ins like `&'a T`.

# Alternatives

* Do not include _output_ lifetime elision for `impl`. Since traits with lifetime
  parameters are quite rare, this would not be a great loss, and would simplify
  the rules somewhat.

* Only add elision rules for `fn`, in keeping with current practice.

* Only add elision for explicit `&` pointers, eliminating one of the drawbacks
  mentioned above. Doing so would impose an ergonomic penalty on abstractions,
  though: `Ref` would be more painful to use than `&`.

# Unresolved questions

The `fn` and `impl` cases tackled above offer the biggest bang for the buck for
lifetime elision. But we may eventually want to consider other opportunities.

## Double lifetimes

Another pattern that sometimes arises is types like `&'a Foo<'a>`. We could
consider an additional elision rule that expands `&Foo` to `&'a Foo<'a>`.

However, such a rule could be easily added later, and it is unclear how common
the pattern is, so it seems best to leave that for a later RFC.

## Lifetime elision in `struct`s

We may want to allow lifetime elision in `struct`s, but the cost/benefit
analysis is much less clear. In particular, it could require chasing an
arbitrary number of (potentially private) `struct` fields to discover the source
of a lifetime parameter for a `struct`. There are also some good reasons to
treat elided lifetimes in `struct`s as `'static`.

Again, since shorthand can be added backwards-compatibly, it seems best to wait.
