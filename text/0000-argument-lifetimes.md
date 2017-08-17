- Feature Name: argument_lifetimes
- Start Date: 2017-08-17
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Improves the clarity, ergonomics, and learnability around explicit lifetimes, so
that instead of writing

```rust
fn two_args<'a, 'b>(arg1: &'a Foo, arg2: &'b Bar) -> &'b Baz
```

you can write:

```rust
fn two_args(arg1: &Foo, arg2: &'b Bar) -> &'b Baz
```

In particular, this RFC completely removes the need for listing lifetime
parameters, instead binding them "in-place" (but with absolute clarity about
*when* this binding is happening):

```rust
fn named_lifetime(arg: &'inner Foo) -> &'inner Bar
fn nested_lifetime(arg: &&'inner Foo) -> &'inner Bar
```

It also proposes linting against leaving off lifetime parameters in structs
(like `Ref` or `Iter`), instead nudging people to use explicit lifetimes in this
case (but leveraging the other improvements to make it ergonomic to do so).

# Motivation
[motivation]: #motivation

Today's system of lifetime elision has a kind of "cliff". In cases where elision
applies (because the necessary lifetimes are clear from the signature), you
don't need to write anything:

```rust
fn one_arg(arg: &Foo) -> &Baz
```

But the moment that lifetimes need to be disambiguated, you suddenly have to
introduce a named lifetime parameter and refer to it throughout, which generally
requires changing three parts of the signature:

```rust
fn two_args<'a, 'b>(arg1: &'a Foo, arg2: &'b Bar) -> &'b Baz
```

In much idiomatic Rust code, these lifetime parameters are given meaningless
names like `'a`, because they're serving merely to tie pieces of the signature
together. This habit indicates a kind of design smell: we're forcing programmers
to conjure up and name a parameter whose identity doesn't matter to them.

Moreover, when reading a signature involving lifetime parameters, you need to
scan the whole thing, keeping `'a` and `'b` in your head, to understand the
pattern of borrowing at play.

These concerns are just a papercut for advanced Rust users, but they also
present a cliff in the learning curve, one affecting the most novel and
difficult to learn part of Rust. In particular, when first explaining borrowing,
we can say that `&` means "borrowed" and that borrowed values coming out of a
function must come from borrowed values in its input:

```rust
fn accessor(&self) -> &Foo
```

It's then not too surprising that when there are multiple input borrows, you
need to disambiguate which one you're borrowing from. But to learn how to do so,
you must learn not only lifetimes, but also the system of lifetime
parameterization and the subtle way you use it to tie lifetimes together. In
the next section, I'll show how this RFC provides a gentler learning curve
around lifetimes and disambiguation.

Another point of confusion for newcomers and old hands alike is the fact that
you can leave lifetimes off when using types:

```rust
struct Iter<'a> { ... }

impl SomeType {
    // Iter here implicitly takes the lifetime from &self
    fn iter(&self) -> Iter { ... }
```

As detailed in the [ergonomics initiative blog post], this bit of lifetime
elision is considered a mistake: it makes it difficult to see at a glance that
borrowing is occurring, especially if you're unfamiliar with the types involved.
This RFC proposes some steps to rectify this situation without regressing
ergonomics significantly.

[ergonomics initiative blog post]: https://blog.rust-lang.org/2017/03/02/lang-ergonomics.html

In short, this RFC seeks to improve the lifetime story for existing and new
users by simultaneously improving clarity and ergonomics. In practice it should
reduce the total occurrences of `<`, `>` and `'a` in signatures, while
*increasing* the overall clarity and explicitness of the lifetime system.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

*Note: this is a **sketch** of what it might look like to teach someone
lifetimes given this RFC**.

## Introducing references and borrowing

*Assume that ownership has already been introduced, but not yet borrowing*.

While ownership is important in Rust, it's not very expressive or convenient by
itself; it's quite common to want to "lend" a value to a function you're
calling, without permanently relinquishing ownership of it.

Rust provides support for this kind of temporary lending through *references*
`&T`, which signify *a temporarily borrowed value of type `T`*. So, for example,
you can write:

```rust
fn print_vec(vec: &Vec<i32>) {
    for i in vec {
        println!("{}", i);
    }
}
```

and you designate lending by writing an `&` on the callee side:

```rust
print_vec(&my_vec)
```

This borrow of `my_vec` lasts only for the duration of the `print_vec` call.

*Imagine more explanation here...*

## Functions that return borrowed data

So far we've only seen functions that *consume* borrowed data; what about
producing it?

In general, borrowed data is always borrowed *from something*. And that thing
must always be available for longer than the borrow is. When a function returns,
its stack frame is destroyed, which means that any borrowed data it returns must
come from outside of its stack frame.

The most typical case is producing new borrowed data from already-borrowed
data. For example, consider a "getter" method:

```rust
struct MyStruct {
    field1: Foo,
    field2: Bar,
}

impl MyStruct {
    fn get_field1(&self) -> &Foo {
        &self.field1
    }
}
```

Here we're making what looks like a "fresh" borrow, it's "derived" from the
existing borrow of `self`, and hence fine to return back to our caller; the
actual `MyStruct` value must live outside our stack frame anyway.

### Pinpointing borrows with lifetimes

For Rust to guarantee safety, it needs to track the *lifetime* of each loan,
which says *for what portion of code the loan is valid*.

In particular, each `&` type also has an associated lifetime---but you can
usually leave it off. The reason is that a lot of code works like the getter
example above, where you're returning borrowed data which could only have come
from the borrowed data you took in. Thus, in `get_field1` the lifetime for
`&self` and for `&Foo` are assumed to be the same.

Rust is conservative about leaving lifetimes off, though: if there's any
ambiguity, you need to say explicitly state the relationships between the
loans. So for example, the following function signature is *not* accepted:

```rust
fn select(data: &Data, params: &Params) -> &Item;
```

Rust cannot tell how long the resulting borrow of `Item` is valid for; it can't
deduce its lifetime. Instead, you need to connect it to one or both of the input
borrows:

```rust
fn select(data: &'data Data, params: &Params) -> &'data Item;
fn select(data: &'both Data, params: &'both Params) -> &'both Item;
```

This notation lets you *name* the lifetime associated with a borrow and refer to
it later:

- In the first variant, we name the `Data` borrow lifetime `'data`, and make
clear that the returned `Item` borrow is valid for the same lifetime.

- In the second variant, we give *both* input lifetimes the *same* name `'both`,
which is a way of asking the compiler to determine their "intersection"
(i.e. the period for which both of the loans are active); we then say the
returned `Item` borrow is valid for that period (which means it may incorporate
data from both of the input borrows).

You can use any names you like when introducing lifetimes (which always start
with `'`), but are encouraged to make them meaningful, e.g. by using the same
name as the parameter.

## `struct`s and lifetimes

Sometimes you need to build data types that contain borrowed data. Since those
types can then be used in many contexts, you can't say in advance what the
lifetime of those borrows will be. Instead, you must take it as a parameter:

```rust
struct VecIter<'vec, T> {
    vec: &'vec Vec<T>,
    index: usize,
}
```

Here we're defining a type for iterating over a vector, without requiring
*ownership* of that vector. To do so, we store a *borrow* of the vector. But
because our new `VecIter` struct contains borrowed data, it needs to surface
that fact, and the lifetime connected with it. It does so by taking an explicit
`'vec` parameter for the relevant lifetime, and using it within.

When using this struct, you can apply explicitly-named lifetimes as usual:

```rust
impl<T> Vec<T> {
    fn iter(&'vec self) -> VecIter<'vec, T> { ... }
}
```

However, in cases like this example, we would normally be able to leave off the
lifetime with `&`, since there's only one source of data we could be borrowing
from. We can do something similar with structs:

```rust
impl<T> Vec<T> {
    fn iter(&self) -> VecIter<_, T> { ... }
}
```

The `_` marker makes clear to the reader that *borrowing is happening*, which
might not otherwise be clear.

## `impl` blocks and lifetimes

When writing an `impl` block for a structure that takes a lifetime parameter,
you can give that parameter a name, but it must begin with an uppercase letter:

```rust
impl<T> VecIter<'Vec, T> { ... }
```

The reason for this distinction is so that there's no potential for confusion
with lifetime names introduced by methods within the `impl` block. In other
words, it makes clear whether the method is *introducing* a new lifetime
(lowercase), or *referring* to the one in the `impl` header (uppercase):

```rust
impl<T> VecIter<'Vec, T> {
    fn foo(&self) -> &'Vec T { ... }
    fn bar(&self, arg: &'arg Bar) -> &'arg Bar { ... }

    // these two are the same:
    fn baz(&self) -> &T { ... }
    fn baz(&'self self) -> &'self T { ... }
}
```

If the type's lifetime is not relevant, you can leave it off using `_`:

```rust
impl<T> VecIter<_, T> { ... }
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## Lifetimes in `impl` headers

When writing an `impl` header, it is deprecated to bind a lifetime parameter
within the generics specification (e.g. `impl<'a>`). It is also deprecated to
use any lifetime variables that do not begin with a capital letter (which is
needed to clearly distinguish lifetimes that are bound in the `impl` header from
those bound in signatures).

Instead the `impl` header can mention lifetimes without adding them as
generics. **These lifetimes *always* become parameters of the `impl`**--after
all, **no lifetime variables can possibly be in scope already**. In other words,
the lifetime parameters of an `impl` are taken to be the set of lifetime
variables it uses.

Thus, where today you would write:

```rust
impl<'a> Iterator for MyIter<'a> { ... }
impl<'a, 'b> SomeTrait<'a> for SomeType<'a, 'b> { ... }
```

tomorrow you would write:

```rust
impl Iterator for MyIter<'A> { ... }
impl SomeTrait<'A> for SomeType<'A, 'B> { ... }
```

## Lifetimes in `fn` signatures

When writing a `fn` declaration, it is deprecated to bind a lifetime parameter
within the generics specification (e.g. `fn foo<'a>(arg: &'a str)`).

Instead:

- If a lowercase lifetime variable occurs anywhere in the signature, it is
  *always bound* by the function (as if it were in the `<>` bindings).
- If an uppercase lifetime variable occurs, it is *always a reference* to a
  lifetime bound by the immediate containing `impl` header, and thus must occur
  in that header.
- It is illegal for the return type to mention any lowercase lifetime variables
  that do not occur in at least one argument.
- As with today's elision rules, lifetimes that appear *only* within `Fn`-style
  bounds or trait object types are bound in higher-rank form (i.e., as if you'd
  written them using a `for<'a>`).

Thus, where today you would write:

```rust
fn elided(&self) -> &str;
fn two_args<'a, 'b>(arg1: &'a Foo, arg2: &'b Bar) -> &'b Baz;

impl<'a> MyStruct<'a> {
    fn foo(&self) -> &'a str;
    fn bar<'b>(&self, arg: &'b str) -> &'b str;
}

fn take_fn<'a>(x: &'a u32, y: for<'b> fn(&'a u32, &'b u32, &'b u32))
```

tomorrow you would write:

```rust
fn elided(&self) -> &str;
fn two_args(arg1: &Foo, arg2: &'b Bar) -> &'b Baz;

impl MyStruct<'A> {
    fn foo(&self) -> &'A str;
    fn bar(&self, arg: &'b str) -> &'b str;
}

fn take_fn(x: &'a u32, y: fn(&'a u32, &'b u32, &'b u32));
```

## The wildcard lifetime

When referring to a type (other than `&`/`&mut`) that requires lifetime
arguments, it is deprecated to leave off those parameters.

Instead, you can write a `_` for the parameters, rather than giving a lifetime
name, which will have identical behavior to leaving them off today.

Thus, where today you would write:

```rust
fn foo(&self) -> Ref<SomeType>
fn iter(&self) -> Iter<T>
```

tomorrow you would write:

```rust
fn foo(&self) -> Ref<_, SomeType>
fn iter(&self) -> Iter<_, T>
```

# Drawbacks
[drawbacks]: #drawbacks

The deprecations here involve some amount of churn (largely in the form of
deleting lifetimes from `<>` blocks, but also sometimes changing their
case). Users exercise a lot of control over when they address that churn and can
do so incrementally. Moreover, we can and should consider addressing this with a
`rustfix`, which should be easy and highly reliable.

The fact that lifetime parameters are not bound in an out-of-band way is
somewhat unusual and might be confusing---but then, so are lifetime parameters!
Putting the bindings out of band buys us very little, as argued in the next
section.

Introducing a case distinction for lifetime names is a bit clunky, and of course
not all spoken languages *have* a case distinction (but for those, another
convention could be applied).

Requiring a `_` rather than being able to leave off lifetimes altogether may be
a slight decrease in ergonomics in some cases.

Cases where you could write `fn foo<'a, 'b: 'a>(...)` now need the `'b: 'a` to
be given in a `where` clause, which might be slightly more verbose. These are
relatively rare, though, due to our type well-formedness rule.

Otherwise, it's a bit hard to see drawbacks here: nothings is made more explicit
or harder to determine, since the binding structure continues to be completely
unambiguous; ergonomics and, arguably, learnability both improve. And
signatures become less noisy and easier to read.

# Rationale and Alternatives
[alternatives]: #alternatives

## Core rationale

The key insight of the proposed design is that out-of-band bindings for lifetime
parameters is buying us very little today:

- For free functions, it's completely unnecessary; the only lifetime "in scope"
  is `'static`, so everything else *must* be a parameter.
- For functions within `impl` blocks, it is solely serving the purpose of
  distinguishing between lifetimes bound by the `impl` header and those bounds
  by the `fn`.

**Thus, if we have some other way of distinguishing between `impl`-bound and
`fn`-bound lifetimes, binding lists for lifetimes are completely redundant**; we
can eliminate them without introducing any ambiguity.

This RFC proposes to make the distinction through case; uppercase lifetimes are
already linted against today. However, we could instead distinguish it purely at
the use-site, for example by writing `outer('a)` or some such to refer to the
`impl` block bindings. In any case, having immediate visual clarity in `fn`
declarations as to whether a lifetime is coming from the `fn` or `impl` block is
a nice benefit.

## Possible extension: "backreferences"

This RFC was written with a particular extension in mind: allowing you to refer
to elided lifetimes through their parameter name, like so:

```rust
fn scramble(&self, arg: &Foo) -> &'self Bar
```

Here, we are referring to `'self` in the return type without any references in
the argument---something that the RFC specifies will be an error. The idea is
that each parameter that involves a single, elided lifetime will be understood
to *bind* a lifetime using that parameter's name.

For the sake of conservative, incremental progress, this RFC punts on the
extension, but tries to leave the door open to it. That will allow us to gather
more data before deciding on this additional step. (To be fully
forward-compatible, we probably need to make it an error to use the name of an
argument as a lifetime *unless* it is used in that argument as well.)

Alternatively, we could consider including this extension up front, perhaps as a
feature gate, so that we can gain experience more quickly, but make
stabilization decisions separately.

## Alternatives

We could consider *only* allowing "backreferences", and otherwise keeping
binding as-is. However, that would forgo the benefits of eliminating out-of-band
binding, which would still be needed in some cases.

We could consider using this as an opportunity to eliminate `'` altogether, by
tying these improvements to a new way of providing lifetimes, e.g. `&ref(x) T`.

The [internals thread] on this topic covers a wide array of syntactic options
for leaving off a struct lifetime (which is `_` in this RFC), including: `'_`,
`&`, `ref`. The choice of `_` was informed by a few things:

- It's short, evocative, and not too visually jarring.
- In signatures, it *always* means "lifetime elided", so there's a clear signal
  re: borrowing.

[internals thread]: (https://internals.rust-lang.org/t/lang-team-minutes-elision-2-0/5182)

As mentioned above, we could consider alternatives to the case distinction in
lifetime variables, instead using something like `outer('a)` to refer to
lifetimes from an `impl` header.

# Unresolved questions
[unresolved]: #unresolved-questions

- Should we include the "backreference" extension up front? It seems very likely
  to be desired in this setup.

- Should we go further and eliminate the need for `for<'a>` notation as well?
