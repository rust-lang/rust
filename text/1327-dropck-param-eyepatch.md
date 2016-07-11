- Feature Name: dropck_eyepatch, generic_param_attrs
- Start Date: 2015-10-19
- RFC PR: [rust-lang/rfcs#1327](https://github.com/rust-lang/rfcs/pull/1327)
- Rust Issue: [rust-lang/rust#34761](https://github.com/rust-lang/rust/issues/34761)

# Summary
[summary]: #summary

Refine the unguarded-escape-hatch from [RFC 1238][] (nonparametric
dropck) so that instead of a single attribute side-stepping *all*
dropck constraints for a type's destructor, we instead have a more
focused system that specifies exactly which type and/or lifetime
parameters the destructor is guaranteed not to access.

Specifically, this RFC proposes adding the capability to attach
attributes to the binding sites for generic parameters (i.e. lifetime
and type paramters). Atop that capability, this RFC proposes adding a
`#[may_dangle]` attribute that indicates that a given lifetime or type
holds data that must not be accessed during the dynamic extent of that
`drop` invocation.

As a side-effect, enable adding attributes to the formal declarations
of generic type and lifetime parameters.

The proposal in this RFC is intended as a *temporary* solution (along
the lines of `#[fundamental]` and *will not* be stabilized
as-is. Instead, we anticipate a more comprehensive approach to be
proposed in a follow-up RFC.

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

First off: The proposal in this RFC is intended as a *temporary*
solution (along the lines of `#[fundamental]` and *will not* be
stabilized as-is. Instead, we anticipate a more comprehensive approach
to be proposed in a follow-up RFC.

Having said that, here is the proposed short-term solution:

 1. Add the ability to attach attributes to syntax that binds formal
    lifetime or type parmeters. For the purposes of this RFC, the only
    place in the syntax that requires such attributes are `impl`
    blocks, as in `impl<T> Drop for Type<T> { ... }`

 2. Add a new fine-grained attribute, `may_dangle`, which is attached
    to the binding sites for lifetime or type parameters on an `Drop`
    implementation.
    This RFC will sometimes call this attribute the "eyepatch",
    since it does
    not make dropck totally blind; just blind on one "side".
 
 3. Add a new requirement that any `Drop` implementation that uses the
    `#[may_dangle]` attribute must be declared as an `unsafe impl`.
    This reflects the fact that such `Drop` implementations have
    an additional constraint on their behavior (namely that they cannot
    access certain kinds of data) that will not be verified by the
    compiler and thus must be verified by the programmer.

 4. Remove `unsafe_destructor_blind_to_params`, since all uses of it
    should be expressible via `#[may_dangle]`.

## Attributes on lifetime or type parameters

This is a simple extension to the syntax.

It is guarded by the feature gate `generic_param_attrs`.

Constructions like the following will now become legal.

Example of eyepatch attribute on a single type parameter:
```rust
unsafe impl<'a, #[may_dangle] X, Y> Drop for Foo<'a, X, Y> {
    ...
}
```

Example of eyepatch attribute on a lifetime parameter:
```rust
unsafe impl<#[may_dangle] 'a, X, Y> Drop for Bar<'a, X, Y> {
    ...
}
```

Example of eyepatch attribute on multiple parameters:
```rust
unsafe impl<#[may_dangle] 'a, X, #[may_dangle] Y> Drop for Baz<'a, X, Y> {
    ...
}
```

These attributes are only written next to the formal binding
sites for the generic parameters. The *usage* sites, points
which refer back to the parameters, continue to disallow the use
of attributes.

So while this is legal syntax:

```rust
unsafe impl<'a, #[may_dangle] X, Y> Drop for Foo<'a, X, Y> {
    ...
}
```

the follow would be illegal syntax (at least for now):

```rust
unsafe impl<'a, X, Y> Drop for Foo<'a, #[may_dangle] X, Y> {
    ...
}
```


## The "eyepatch" attribute

Add a new attribute, `#[may_dangle]` (the "eyepatch").

It is guarded by the feature gate `dropck_eyepatch`.

The eyepatch is similar to `unsafe_destructor_blind_to_params`: it is
part of the `Drop` implementation, and it is meant
to assert that a destructor is guaranteed not to access certain kinds
of data accessible via `self`.

The main difference is that the eyepatch is applied to a single
generic parameter: `#[may_dangle] ARG`.
This specifies exactly *what*
the destructor is blind to (i.e., what will dropck treat as
inaccessible from the destructor for this type).

There are two things one can supply as the `ARG` for a given eyepatch: 
one of the type parameters for the type,
or one of the lifetime parameters
for the type.

When used on a type, e.g. `#[may_dangle] T`, the programmer is
asserting the only uses of values of that type will be to move or drop
them. Thus, no fields will be accessed nor methods called on values of
such a type (apart from any access performed by the destructor for the
type when the values are dropped). This ensures that no dangling
references (such as when `T` is instantiated with `&'a u32`) are ever
accessed in the scenario where `'a` has the same lifetime as the value
being currently destroyed (and thus the precise order of destruction
between the two is unknown to the compiler).

When used on a lifetime, e.g. `#[may_dangle] 'a`, the programmer is
asserting that no data behind a reference of lifetime `'a` will be
accessed by the destructor. Thus, no fields will be accessed nor
methods called on values of type `&'a Struct`, ensuring that again no
dangling references are ever accessed by the destructor.

## Require `unsafe` on Drop implementations using the eyepatch

The final detail is to add an additional check to the compiler
to ensure that any use of `#[may_dangle]` on a `Drop` implementation
imposes a requirement that that implementation block use
`unsafe impl`.<sup>[2](#footnote1)</sup>

This reflects the fact that use of `#[may_dangle]` is a
programmer-provided assertion about the behavior of the `Drop`
implementation that must be valided manually by the programmer.
It is analogous to other uses of `unsafe impl` (apart from the
fact that the `Drop` trait itself is not an `unsafe trait`).

### Examples adapted from the Rustonomicon

[nomicon dropck]: https://doc.rust-lang.org/nightly/nomicon/dropck.html

So, adapting some examples from the Rustonomicon
[Drop Check][nomicon dropck] chapter, we would be able to write
the following.

Example of eyepatch on a lifetime parameter::

```rust
struct InspectorA<'a>(&'a u8, &'static str);

unsafe impl<#[may_dangle] 'a> Drop for InspectorA<'a> {
    fn drop(&mut self) {
        println!("InspectorA(_, {}) knows when *not* to inspect.", self.1);
    }
}
```

Example of eyepatch on a type parameter:

```rust
use std::fmt;

struct InspectorB<T: fmt::Display>(T, &'static str);

unsafe impl<#[may_dangle] T: fmt::Display> Drop for InspectorB<T> {
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
unsafe impl<#[may_dangle]T, A:Allocator> Drop for RawVec<T, A> {
    /// Frees the memory owned by the RawVec *without* trying to Drop its contents.
    fn drop(&mut self) {
        [... free memory using self.alloc ...]
    }
}
```

The use of `#[may_dangle] T` here asserts that even
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

unsafe impl<#[may_dangle] 'a, 'b, #[may_dangle] 'c> Drop for InspectorC<'a,'b,'c> {
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
[implementation][pnkfelix prototype] of the `#[may_dangle]` attribute.
It uses the substitution machinery we already have in the compiler
to express the semantics above.

## Limitations of prototype (not part of design)

Here we note a few limitations of the current prototype. These
limitations are *not* being proposed as part of the specification of
the feature.

<a name="footnote1">2.</a> The compiler does not yet enforce (or even
allow) the use of `unsafe impl` for `Drop` implementations that use
the `#[may_dangle]` attribute.

Fixing the above limitations should just be a matter of engineering,
not a fundamental hurdle to overcome in the feature's design in the
context of the language.

[pnkfelix prototype]: https://github.com/pnkfelix/rust/commits/dropck-eyepatch

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

[RFC 1238 alternatives]: https://github.com/rust-lang/rfcs/blob/master/text/1238-nonparametric-dropck.md#continue-supporting-parametricity

# Alternatives
[alternatives]: #alternatives

Note: The alternatives section for this RFC is particularly
note-worthy because the ideas here may serve as the basis for a more
comprehensive long-term approach.

## Make dropck "see again" via (focused) where-clauses

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

  * Since the eyepatch attribute is to be limited to type and lifetime
    parameters, this approach is more expressive,
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
    replace occurrences of `#[may_dangle] T` with `T: ?Special`
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

Is the definition of the drop-check rule sound with this `patched(D)`
variant?  (We have not proven any previous variation of the rule
sound; I think it would be an interesting student project though.)
