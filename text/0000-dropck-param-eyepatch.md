- Feature Name: dropck_eyepatch
- Start Date: 2015-10-19
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Refine the unguarded-escape-hatch from [RFC 1238][] (nonparametric
dropck) so that instead of a single attribute side-stepping *all*
dropck constraints for a type's destructor, we instead have a more
focused attribute that specifies exactly which type and/or lifetime
parameters the destructor is guaranteed not to access.

[RFC 1238]: https://github.com/rust-lang/rfcs/blob/master/text/1238-nonparametric-dropck.md
[RFC 769]: https://github.com/rust-lang/rfcs/blob/master/text/0769-sound-generic-drop.md

# Motivation
[motivation]: #motivation

The unguarded escape hatch (UGEH) from [RFC 1238] is a blunt
instrument: when you use `unsafe_destructor_blind_to_params`, it is
asserting that your destructor does not access borrowed data whose
type includes *any* lifetime or type parameter of the type.

For example, the current destructor for `RawVec<T>` (in `liballoc/`)
looks like this:

```rust
impl<T> Drop for RawVec<T> {
    #[unsafe_destructor_blind_to_params]
    /// Frees the memory owned by the RawVec *without* trying to Drop its contents.
    fn drop(&mut self) {
        [... free memory using global system allocator ...]
    }
}
```

The above is sound today, because the above destructor does not call
any methods that can access borrowed data in the values of type `T`,
and so we do not need to enforce the drop-ordering constraints imposed
when you leave out the `unsafe_destructor_blind_to_params` attribute.

While the above attribute suffices for many use cases today, it is not
fine-grain enough for other cases of interest. In particular, it
cannot express that the destructor will not access borrowed data
behind a *subset* of the type parameters.

Here are two concrete examples of where the need for this arises:

## Example: `CheckedHashMap`

The original Sound Generic Drop proposal ([RFC 769][])
had an [appendix][RFC 769 CheckedHashMap] with an example of a
`CheckedHashMap<K, V>` type that called the hashcode method
for all of the keys in the map in its destructor.
This is clearly a type where we *cannot* claim that we do not access
borrowed data potentially hidden behind `K`, so it would be unsound
to use the blunt `unsafe_destructor_blind_to_params` attribute on this
type.

However, the values of the `V` parameter to `CheckedHashMap` are, in
all likelihood, *not* accessed by the `CheckedHashMap` destructor. If
that is the case, then it should be sound to instantiate `V` with a
type that contains references to other parts of the map (e.g.,
references to the keys or to other values in the map). However, we
cannot express this today: There is no way to say that the
`CheckedHashMap` will not access borrowed data that is behind *just*
`V`.

[RFC 769 CheckedHashMap]: https://github.com/rust-lang/rfcs/blob/master/text/0769-sound-generic-drop.md#appendix-a-why-and-when-would-drop-read-from-borrowed-data

## Example: `Vec<T, A:Allocator=DefaultAllocator>`

The Rust developers have been talking for [a long time][RFC Issue 538]
about adding an `Allocator` trait that would allow users to override
the allocator used for the backing storage of collection types like
`Vec` and `HashMap`.

For example, we would like to generalize the `RawVec` given above as
follows:

```rust
#[unsafe_no_drop_flag]
pub struct RawVec<T, A:Allocator=DefaultAllocator> {
    ptr: Unique<T>,
    cap: usize,
    alloc: A,
}

impl<T, A:Allocator> Drop for RawVec<T, A> {
    #[should_we_put_ugeh_attribute_here_or_not(???)]
    /// Frees the memory owned by the RawVec *without* trying to Drop its contents.
    fn drop(&mut self) {
        [... free memory using self.alloc ...]
    }
}
```

However, we *cannot* soundly add an allocator parameter to a
collection that today uses the `unsafe_destructor_blind_to_params`
UGEH attribute in the destructor that deallocates, because that blunt
instrument would allow someone to write this:

```rust
// (`ArenaAllocator`, when dropped, automatically frees its allocated blocks)

// (Usual pattern for assigning same extent to `v` and `a`.)
let (v, a): (Vec<Stuff, &ArenaAllocator>, ArenaAllocator);

a = ArenaAllocator::new();
v = Vec::with_allocator(&a);

... v.push(stuff) ...

// at end of scope, `a` may be dropped before `v`, invalidating
// soundness of subsequent invocation of destructor for `v` (because
// that would try to free buffer of `v` via `v.buf.alloc` (== `&a`)).
```

The only way today to disallow the above unsound code would be to
remove `unsafe_destructor_blind_to_params` from `RawVec`/ `Vec`, which
would break other code (for example, code using `Vec` as the backing
storage for [cyclic graph structures][dropck_legal_cycles.rs]).

[RFC Issue 538]: https://github.com/rust-lang/rfcs/issues/538

[dropck_legal_cycles.rs]: https://github.com/rust-lang/rust/blob/098a7a07ee6d11cf6d2b9d18918f26be95ee2f66/src/test/run-pass/dropck_legal_cycles.rs

# Detailed design
[detailed design]: #detailed-design

 1. Add a new fine-grained attribute, `unsafe_destructor_blind_to`
    (which this RFC will sometimes call the "eyepatch", since it does
    not make dropck totally blind; just blind on one "side").

 2. Remove `unsafe_destructor_blind_to_params`, since all uses of it
    should be expressible via `unsafe_destructor_blind_to` (once that
    has been completely implemented).

## The "eyepatch" attribute

Add a new attribute, `unsafe_destructor_blind_to(ARG)` (the "eyepatch").

The eyepatch is similar to `unsafe_destructor_blind_to_params`: it is
attached to the destructor<sup>[1](#footnote1)</sup>, and it is meant
to assert that a destructor is guaranteed not to access certain kinds
of data accessible via `self`.

The main difference is that the eyepatch has a single required
parameter, `ARG`. This is the place where you specify exactly *what*
the destructor is blind to (i.e., what will dropck treat as
inaccessible from the destructor for this type).

There are two things one can put the `ARG` for a given eyepatch: one
of the type parameters for the type, or one of the lifetime parameters
for the type.<sup>[2](#footnote2)</sup>

### Examples stolen from the Rustonomicon

[nomicon dropck]: https://doc.rust-lang.org/nightly/nomicon/dropck.html

So, adapting some examples from the Rustonomicon
[Drop Check][nomicon dropck] chapter, we would be able to write
the following.

Example of eyepatch on a lifetime parameter::

```rust
struct InspectorA<'a>(&'a u8, &'static str);

impl<'a> Drop for InspectorA<'a> {
    #[unsafe_destructor_blind_to('a)]
    fn drop(&mut self) {
        println!("InspectorA(_, {}) knows when *not* to inspect.", self.1);
    }
}
```

Example of eyepatch on a type parameter:

```rust
use std::fmt;

struct InspectorB<T: fmt::Display>(T, &'static str);

impl<T: fmt::Display> Drop for InspectorB<T> {
    #[unsafe_destructor_blind_to(T)]
    fn drop(&mut self) {
        println!("InspectorB(_, {}) knows when *not* to inspect.", self.1);
    }
}
```

Both of the above two examples are much the same as if we had used the
old `unsafe_destructor_blind_to_params` UGEH attribute.

### Example: RawVec

To generalize `RawVec` from the [motivation](#motivation) with an
`Allocator` correctly (that is, soundly and without breaking existing
code), we would now write:

```rust
impl<T, A:Allocator> Drop for RawVec<T, A> {
    #[unsafe_destructor_blind_to(T)]
    /// Frees the memory owned by the RawVec *without* trying to Drop its contents.
    fn drop(&mut self) {
        [... free memory using self.alloc ...]
    }
}
```

The use of `unsafe_destructor_blind_to(T)` here asserts that even
though the destructor may access borrowed data through `A` (and thus
dropck must impose drop-ordering constraints for lifetimes occurring
in the type of `A`), the developer is guaranteeing that no access to
borrowed data will occur via the type `T`.

The latter is not expressible today even with
`unsafe_destructor_blind_to_params`; there is no way to say that a
type will not access `T` in its destructor while also ensuring the
proper drop-ordering relationship between `RawVec<T, A>` and `A`.

### Example; Multiple Lifetimes

Example: The above `InspectorA` carried a `&'static str` that was
always safe to access from the destructor.

If we wanted to generalize this type a bit, we might write:

```rust
struct InspectorC<'a,'b,'c>(&'a str, &'b str, &'c str);

impl<'a,'b,'c> Drop for InspectorC<'a,'b,'c> {
    #[unsafe_destructor_blind_to('a)]
    #[unsafe_destructor_blind_to('c)]
    fn drop(&mut self) {
        println!("InspectorA(_, {}, _) knows when *not* to inspect.", self.1);
    }
}
```

This type, like `InspectorA`, is careful to only access the `&str`
that it holds in its destructor; but now the borrowed string slice
does not have `'static` lifetime, so we must make sure that we do not
claim that we are blind to its lifetime (`'b`).

(This example also illustrates that one can attach multiple instances
of the eyepatch attribute to a destructor, each with a distinct input
for its `ARG`.)

Given the definition above, this code will compile and run properly:

```rust
fn this_will_work() {
    let b; // ensure that `b` strictly outlives `i`.
    let (i,a,c);
    a = format!("a");
    b = format!("b");
    c = format!("c");
    i = InspectorC(a, b, c);
}
```

while this code will be rejected by the compiler:

```rust
fn this_will_not_work() {
    let (a,c);
    let (i,b); // OOPS: `b` not guaranteed to survive for `i`'s destructor.
    a = format!("a");
    b = format!("b");
    c = format!("c");
    i = InspectorC(a, b, c);
}
```

## Semantics

How does this work, you might ask?

The idea is actually simple: the dropck rule stays mostly the same,
except for a small twist.

The Drop-Check rule at this point essentially says:

> if the type of `v` owns data of type `D`, where
>
>  (1.) the `impl Drop for D` is either type-parametric, or lifetime-parametric over `'a`, and
>  (2.) the structure of `D` can reach a reference of type `&'a _`,
>
> then `'a` must strictly outlive the scope of `v`

The main change we want to make is to the second condition.
Instead of just saying "the structure of `D` can reach a reference of type `&'a _`",
we want first to replace eyepatched lifetimes and types within `D` with `'static` and `()`,
respectively. Call this revised type `patched(D)`.

Then the new condition is:

>  (2.) the structure of patched(D) can reach a reference of type `&'a _`,

*Everything* else is the same.

In particular, the patching substitution is *only* applied with
respect to a particular destructor. Just because `Vec<T>` is blind to `T`
does not mean that we will ignore the actual type instantiated at `T`
in terms of drop-ordering constraints.

For example, in `Vec<InspectorC<'a,'name,'c>>`, even though `Vec`
itself is blind to the whole type `InspectorC<'a, 'name, 'c>` when we
are considering the `impl Drop for Vec`, we *still* honor the
constraint that `'name` must strictly outlive the `Vec` (because we
continue to consider all `D` that is data owned by a value `v`,
including when `D` == `InspectorC<'a,'name,'c>`).

## Prototype
[prototype]: #prototype

pnkfelix has implemented a proof-of-concept
[implementation][pnkfelix prototype] of this feature.
It uses the substitution machinery we already have in the compiler
to express the semantics above.

## Limitations of prototype (not part of design)

Here we note a few limitations of the current prototype. These
limitations are *not* being proposed as part of the specification of
the feature.

<a name="footnote1">1.</a> The eyepatch is not attached to the
destructor in the current [prototype][pnkfelix prototype]; it is
instead attached to the `struct`/`enum` definition itself.

<a name="footnote2">2.</a> The eyepatch is only able to accept a type
parameter, not a lifetime, in the current
[prototype][pnkfelix prototype]; it is instead attached to the
`struct`/`enum` definition itself.

Fixing the above limitations should just be a matter of engineering,
not a fundamental hurdle to overcome in the feature's design in the
context of the language.

[pnkfelix prototype]: https://github.com/pnkfelix/rust/commits/fsk-nonparam-blind-to-indiv

# Drawbacks
[drawbacks]: #drawbacks

## Ugliness

This attribute, like the original `unsafe_destructor_blind_to_params`
UGEH attribute, is ugly.

## Unchecked assertions boo

It would be nicer if to actually change the language in a way where we
could check the assertions being made by the programmer, rather than
trusting them. (pnkfelix has some thoughts on this, which are mostly
reflected in what he wrote in the [RFC 1238 alternatives][].)

## Attributes lack hygiene
[attributes-lack-hygiene]: #attributes-lack-hygiene

As noted by arielb1, putting type parameter identifiers into attributes
is not likely to play well with macro hygiene.

Here is a concrete example:

```rust
struct Yell2<A:Debug,B:Debug>(A, B);

macro_rules! make_yell2a {
    ($A:ident, $B:ident) => {
        impl<$A:Debug,$B:Debug> Drop for Yell2<$A,$B> {
            #[unsafe_destructor_blind_to(???)]  // <----
            fn drop(&mut self) {
                println!("Yell1(_, {:?})", self.1);
            }
        }
    }
}

make_yell2a!(X, Y);
```

Here is the issue: In the above, what does one put in for the `???` to
say that we are blind to the first type parameter to `Yell2`?
`#[unsafe_destructor_blind_to(A)` would be nonsense, becauase in the instantiation of the macro, `$A` will be mapped to the identifier `X`.  so perhaps we should write it is blind to `X` -- but to me one big point of macro hygiene is that a macro definition should not have to build in knowledge of the identifiers chosen at the usage site, and this is the opposite of that.

(I don't think `#[unsafe_destructor_blind_to($A)` works, because our attribute system operates at the same meta-level that macros operate at , but I would be happy to be proven wrong.)

----

Despite my somewhat dire attitude above, I don't think this is a significant problem in the short term. This sort of macro is probably rare, and the combination of this macro with UGEH is doubly so. You cannot define a destructor multiple times for the same type, so it seems weird to me to abstract this code construction at this particular level.


[RFC 1238 alternatives]: https://github.com/rust-lang/rfcs/blob/master/text/1238-nonparametric-dropck.md#continue-supporting-parametricity

# Alternatives
[alternatives]: #alternatives

## unsafe_destructor_blind_to(T1, T2, ...)

The eyepatch could take multiple arguments, rather than requiring a
distinct instance of the attribute for each parameter that we are
blind to.

However, I think that each usage of the attribute needs to be
considered, since it represents a separate "attack vector" where
unsoundness can be introduced, and therefore it deserves more than
just a comma and a space added to the program text when it is added.

(I only weakly support the latter position; it is obviously easy
to support this form if that is deemed desirable.)

## Use a blacklist not a whitelist
[blacklist-not-whitelist]: #use-a-blacklist-not-a-whitelist

The `unsafe_destructor_blind_to` attribute acts as a whitelist of
parameters that we are telling dropck to ignore in its analysis
of this destructor.

We could instead add a way to list the lifetimes and/or
type-expressions (e.g. parameters, projections from parameters) that
the destructor may access (and thus treat that list as a blacklist of
parameters that dropck needs to *include* in its analysis).

arielb1 first suggested this as an attribute form
[here][blacklist attribute], but then provided a different formulation
of the idea by expressing it as a [`where`-clause][blacklist where] on
the `fn drop` method (which is what I will show in the next section).

[blacklist attribute]: https://github.com/rust-lang/rfcs/pull/1327#issuecomment-149302743

[blacklist where]: https://github.com/rust-lang/rfcs/pull/1327#issuecomment-149329351

## Make dropck "see again" via (focused) where-clauses

(This alternative carries over some ideas from
[the previous section][blacklist-not-whitelist], but it stands well on
its own as something to consider, so I am giving it its own section.)

The idea is that we keep the UGEH attribute, blunt hammer that it is.
You first opt out of the dropck ordering constraints via that, and
then you add back in ordering constraints via `where` clauses.

(The ordering constraints in question would normally be *implied* by
the dropck analysis; the point is that UGEH is opting out of that
analysis, and so we are now adding them back in.)

Here is the allocator example expressed in this fashion:

```rust
impl<T, A:Allocator> Drop for RawVec<T, A> {
    #[unsafe_destructor_blind_to_params]
    /// Frees the memory owned by the RawVec *without* trying to Drop its contents.
    fn drop<'s>(&'s mut self) where A: 's {
    //                        ~~~~~~~~~~~
    //                             |
    //                             |
    // This constraint (that `A` outlives `'s`), and other conditions
    // relating `'s` and `Self` are normally implied by Rust's type
    // system, but `unsafe_destructor_blind_to_params` opts out of
    // enforcing them. This `where`-clause is opting back into *just*
    // the `A:'s` again.
    //
    // Note we are *still* opting out of `T: 's` via
    // `unsafe_destructor_blind_to_params`, and thus our overall
    // goal (of not breaking code that relies on `T` not having to
    // survive the destructor call) is accomplished.

        [... free memory using self.alloc ...]
    }
}
```

This approach, if we can make it work, seems fine to me. It certainly
avoids a number of problems that the eyepatch attribute has.

Advantages of fn-drop-with-where-clauses:

  * It completely sidesteps the [hygiene issue][attributes-lack-hygiene].

  * If the eyepatch attribute is to be limited to identifiers (type
    parameters) and lifetimes, then this approach is more expressive,
    since it would allow one to put type-projections into the
    constraints.

Drawbacks of fn-drop-with-where-clauses:

  * Its not 100% clear what our implementation strategy will be for it,
    while the eyepatch attribute does have a [prototype].

    I actually do not give this drawback much weight; resolving this
    may be merely a matter of just trying to do it: e.g., build up the
    set of where-clauses when we make the ADT's representatin, and
    then have `dropck` insert instantiate and insert them as needed.

  * It might have the wrong ergonomics for developers: It seems bad to
    have the blunt hammer introduce all sorts of potential
    unsoundness, and rely on the developer to keep the set of
    `where`-clauses on the `fn drop` up to date.

    This would be a pretty bad drawback, *if* the language and
    compiler were to stagnate. But my intention/goal is to eventually
    put in a [sound compiler analysis][wait-for-proper-parametricity].
    In other words, in the future, I will be more concerned about the
    ergonomics of the code that uses the sound analysis. I will not be
    concerned about "gotcha's" associated with the UGEH escape hatch.

(The most important thing I want to convey is that I believe that both
the eyepatch attribute and fn-drop-with-where-clauses are capable of
resolving the real issues that I face today, and I would be happy for
either proposal to be accepted.)

## Wait for proper parametricity
[wait-for-proper-parametricity]: #wait-for-proper-parametricity

As alluded to in the [drawbacks][], in principle we could provide
similar expressiveness to that offered by the eyepatch (which is
acting as a fine-grained escape hatch from dropck) by instead offering
some language extension where the compiler would actually analyze the
code based on programmer annotations indicating which types and
lifetimes are not used by a function.

In my opinion I am of two minds on this (but they are both in favor
this RFC rather than waiting for a sound compiler analysis):

 1. We will always need an escape hatch. The programmer will always need
    a way to assert something that she knows to be true, even if the compiler
    cannot prove it. (A simple example: Calling a third-party API that has not
    yet added the necessary annotations.)

    This RFC is proposing that we keep an escape hatch, but we make it more
    expressive.

 2. If we eventually *do* have a sound compiler analysis, I see the
    compiler changes and library annotations suggested by this RFC as
    being in line with what that compiler analysis would end up using
    anyway. In other words: Assume we *did* add some way for the programmer
    to write that `T` is parametric (e.g. `T: ?Special` in the [RFC 1238 alternatives]).
    Even then, we would still need the compiler changes suggested by this RFC,
    and at that point hopefully the task would be for the programmer to mechanically
    replace occurrences of `#[unsafe_destructor_blind_to(T)` with `T: ?Special`
    (and then see if the library builds).

    In other words, I see the form suggested by this RFC as being a step *towards*
    a proper analysis, in the sense that it is getting programmers used to thinking
    about the individual parameters and their relationship with the container, rather
    than just reasoning about the container on its own without any consideration
    of each type/lifetime parameter.

## Do nothing

If we do nothing, then we cannot add `Vec<T, A:Allocator>` soundly.

# Unresolved questions
[unresolved]: #unresolved-questions

Is there any issue with writing `'a` in an attribute like
`#[unsafe_destructor_blind_to('a)]`?  (The prototype, as mentioned
[above](#footnote2), does not currently accept lifetime parameter
inputs, so I do not know the answer off hand.

Is the definition of the drop-check rule sound with this `patched(D)`
variant?  (We have not proven any previous variation of the rule
sound; I think it would be an interesting student project though.)
