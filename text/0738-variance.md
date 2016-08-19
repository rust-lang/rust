- Start Date: 2014-12-19
- RFC PR: https://github.com/rust-lang/rfcs/pull/738
- Rust Issue: https://github.com/rust-lang/rust/issues/22212

# Summary

- Use inference to determine the *variance* of input type parameters.
- Make it an error to have unconstrained type/lifetime parameters.
- Revamp the variance markers to make them more intuitive and less numerous.
  In fact, there are only two: `PhantomData` and `PhantomFn`.
- Integrate the notion of `PhantomData` into other automated compiler
  analyses, notably OIBIT, that can otherwise be deceived into yielding
  incorrect results.

# Motivation

## Why variance is good

Today, all type parameters are invariant. This can be problematic
around lifetimes. A particular common example of where problems
arise is in the use of `Option`. Here is a simple example. Consider
this program, which has a struct containing two references:

```
struct List<'l> {
    field1: &'l int,
    field2: &'l int,
}

fn foo(field1: &int, field2: &int) {
    let list = List { field1: field1, field2: field2 };
    ...
}

fn main() { }
```

Here the function `foo` takes two references with distinct lifetimes.
The variable `list` winds up being instantiated with a lifetime that
is the intersection of the two (presumably, the body of `foo`).  This
is good.

If we modify this program so that one of those references is optional,
however, we will find that it gets a compilation error:

```
struct List<'l> {
    field1: &'l int,
    field2: Option<&'l int>,
}

fn foo(field1: &int, field2: Option<&int>) {
    let list = List { field1: field1, field2: field2 };
        // ERROR: Cannot infer an appropriate lifetime
    ...
}

fn main() { }
```

The reason for this is that because `Option` is *invariant* with
respect to its argument type, it means that the lifetimes of `field1`
and `field2` must match *exactly*. It is not good enough for them to
have a common subset. This is not good.

## What variance is

[Variance][v] is a general concept that comes up in all languages that
combine subtyping and generic types. However, because in Rust all
subtyping is related to the use of lifetimes parameters, Rust uses
variance in a very particular way. Basically, variance is a
determination of when it is ok for lifetimes to be approximated
(either made bigger or smaller, depending on context).

Let me give a few examples to try and clarify how variance works.
Consider this simple struct `Context`:

```rust
struct Context<'data> {
    data: &'data u32,
    ...
}
```

Here the `Context` struct has one lifetime parameter, `data`, that
represents the lifetime of some data that it references. Now let's
imagine that the lifetime of the data is some lifetime we call
`'x`. If we have a context `cx` of type `Context<'x>`, it is ok to
(for example) pass `cx` as an argment where a value of type
`Context<'y>` is required, so long as `'x : 'y` ("`'x` outlives
`'y`"). That is, it is ok to approximate `'x` as a shorter lifetime
like `'y`. This makes sense because by changing `'x` to `'y`, we're
just pretending the data has a shorter lifetime than it actually has,
which can't do any harm.  Here is an example:

```rust
fn approx_context<'long,'short>(t: &Context<'long>, data: &'short Data)
    where 'long : 'short
{
    // here we approximate 'long as 'short, but that's perfectly safe.
    let u: &Context<'short> = t;
    do_something(u, data)
}

fn do_something<'x>(t: &Context<'x>, data: &'x Data) {
   ...
}
```

This case has been traditionally called "contravariant" by Rust,
though some argue (somewhat persuasively) that
["covariant" is the better terminology][391].  In any case, this RFC
generally abandons the "variance" terminology in publicly exposed APIs
and bits of the language, making this a moot point (in this RFC,
however, I will stick to calling lifetimes which may be made smaller
"contravariant", since that is what we have used in the past).

[391]: https://github.com/rust-lang/rfcs/issues/391

Next let's consider a struct with interior mutability:

```rust
struct Table<'arg> {
    cell: Cell<&'arg Foo>
}
```

In the case of `Table`, it is not safe for the compiler to approximate
the lifetime `'arg` at all. This is because `'arg` appears in a
mutable location (the interior of a `Cell`). Let me show you what
could happen if we did allow `'arg` to be approximated:

```rust
fn innocent<'long>(t: &Table<'long>) {
    {
        let foo: Foo = ..;
        evil(t, &foo);
    }
    t.cell.get() // reads `foo`, which has been destroyed
}

fn evil<'long,'short>(t: &Table<'long>, s: &'short Foo)
    where 'long : 'short
{
    // The following assignment is not legal, but it would be legal
    let u: &Table<'short> = t;
    u.cell.set(s);
}
```

Here the function `evil()` changes contents of `t.cell` to point at
data with a shorter lifetime than `t` originally had. This is bad
because the caller still has the old type (`Table<'long>`) and doesn't
know that data with a shorter lifetime has been inserted.  (This is
traditionally called "invariant".)

Finally, there can be cases where it is ok to make a lifetime
*longer*, but not shorter. This comes up (for example) in a type like
`fn(&'a u8)`, which may be safely treated as a `fn(&'static u8)`.

[v]: http://en.wikipedia.org/wiki/Covariance_and_contravariance_%28computer_science%29

## Why variance should be inferred

Actually, lifetime parameters already have a notion of variance, and
this varinace is fully inferred. In fact, the proper variance for type
parameters is *also* being inferred, we're just largely ignoring
it. (It's not completely ignored; it informs the variance of
lifetimes.)

The main reason we chose inference over declarations is that variance
is rather tricky business. Most of the time, it's annoying to have to
think about it, since it's a purely mechanical thing. The main reason
that it pops up from time to time in Rust today (specifically, in
examples like the one above) is because we *ignore* the results of
inference and just make everything invariant.

But in fact there is another reason to prefer inference. When manually
specifying variance, it is easy to get those manual specifications
wrong. There is one example later on where the author did this, but
using the mechanisms described in this RFC to guide the inference
actually led to the correct solution.

## The corner case: unused parameters and parameters that are only used unsafely

Unfortunately, variance inference only works if type parameters are
actually *used*. Otherwise, there is no data to go on. You might think
parameters would always be used, but this is not true. In particular,
some types have "phantom" type or lifetime parameters that are not
used in the body of the type. This generally occurs with unsafe code:

    struct Items<'vec, T> { // unused lifetime parameter 'vec
        x: *mut T
    }

    struct AtomicPtr<T> { // unused type parameter T
        data: AtomicUint  // represents an atomically mutable *mut T, really
    }

Since these parameters are unused, the inference can reasonably
conclude that `AtomicPtr<int>` and `AtomicPtr<uint>` are
interchangable: after all, there are no fields of type `T`, so what
difference does it make what value it has? This is not good (and in
fact we have behavior like this today for lifetimes, which is a common
source of error).

To avoid this hazard, the RFC proposes to make it an error to have a
type or lifetime parameter whose variance is not constrained. Almost
always, the correct thing to do in such a case is to either remove the
parameter in question or insert a *marker type*. Marker types
basically inform the inference engine to pretend as if the type
parameter were used in particular ways. They are discussed in the next section.

## Revamping the marker types

### The UnsafeCell type

As today, the `UnsafeCell<T>` type is well-known to `rustc` and is
always considered invariant with respect to its type parameter `T`.

### Phantom data

This RFC proposes to replace the existing marker types
(`CovariantType`, `ContravariantLifetime`, etc) with a single type,
`PhantomData`:

```rust
// Represents data of type `T` that is logically present, although the
// type system cannot see it. This type is covariant with respect to `T`.
struct PhantomData<T>;
```

An instance of `PhantomData` is used to represent data that is
logically present, although the type system cannot see
it. `PhantomData` is covariant with respect to its type parameter `T`. Here are
some examples of uses of `PhantomData` from the standard library:

```rust
struct AtomicPtr<T> {
    data: AtomicUint,

    // Act as if we could reach a `*mut T` for variance. This will
    // make `AtomicPtr` *invariant* with respect to `T` (because `T` appears
    // underneath the `mut` qualifier).
    marker: PhantomData<*mut T>,
}

pub struct Items<'a, T: 'a> {
    ptr: *const T,
    end: *const T,

    // Act as if we could reach a slice `[T]` with lifetime `'a`.
    // Induces covariance on `T` and suitable variance on `'a`
    // (covariance using the definition from rfcs#391).
    marker: marker::PhantomData<&'a [T]>,
}
```

Note that `PhantomData` can be used to induce covariance, invariance, or contravariance
as desired:

```rust
PhantomData<T>         // covariance
PhantomData<*mut T>    // invariance, but see "unresolved question"
PhantomData<Cell<T>>   // invariance
PhantomData<fn(T)>     // contravariant
```

Even better, the user doesn't really have to understand the terms
covariance, invariance, or contravariance, but simply to accurately
model the kind of data that the type system should pretend is present.

**Other uses for phantom data.** It turns out that phantom data is an
important concept for other compiler analyses. One example is the
OIBIT analysis, which decides whether certain traits (like `Send` and
`Sync`) are implemented by recursively examining the fields of structs
and enums. OIBIT should treat phantom data the same as normal
fields. Another example is the ongoing work for removing the
`#[unsafe_dtor]` annotation, which also sometimes requires a recursive
analysis of a similar nature.

### Phantom functions

One limitation of the marker type `PhantomData` is that it cannot be
used to constrain unused parameters appearing on traits. Consider
the following example:

```rust
trait Dummy<T> { /* T is never used here! */ }
```

Normally, the variance of a trait type parameter would be determined
based on where it appears in the trait's methods: but in this case
there are no methods. Therefore, we introduce two special traits that
can be used to induce variance. Similarly to `PhantomData`, these
traits represent parts of the interface that are logically present, if
not actually present:

    // Act as if there were a method `fn foo(A) -> R`. Induces contravariance on A
    // and covariance on R.
    trait PhantomFn<A,R>;

These traits should appear in the supertrait list. For example, the
`Dummy` trait might be modified as follows:

```rust
trait Dummy<T> : PhantomFn() -> T { }
```

As you can see, the `()` notation can be used with `PhantomFn` as
well.

### Designating marker traits

In addition to phantom fns, there is a convenient trait `MarkerTrait`
that is intended for use as a supertrait for traits that designate
sets of types. These traits often have no methods and thus no actual
uses of `Self`. The builtin bounds are a good example:

```rust
trait Copy : MarkerTrait { }
trait Sized : MarkerTrait { }
unsafe trait Send : MarkerTrait { }
unsafe trait Sync : MarkerTrait { }
```

`MarkerTrait` is not builtin to the language or specially understood
by the compiler, it simply encapsulates a common pattern. It is
implemented as follows:

```rust
trait MarkerTrait for Sized? : PhantomFn(Self) -> bool { }
impl<Sized? T> MarkerTrait for T { }
```

Intuitively, `MarkerTrait` extends `PhantomFn(Self)` because it is "as
if" the traits were defined like:

```rust
trait Copy {
    fn is_copyable(&self) -> bool { true }
}
```

Here, the type parameter `Self` appears in argument position, which is
contravariant.

**Why contravariance?** To see why contravariance is correct, you have
to consider what it means for `Self` to be contravariant for a marker
trait. It means that if I have evidence that `T : Copy`, then I can
use that as evidence to show that `U
: Copy` if `U <: T`. More formally:

    (T : Copy) <: (U : Copy)   // I can use `T:Copy` where `U:Copy` is expected...
    U <: T                     // ...so long as `U <: T`

More intuitively, it means that if a type `T` implements the marker,
than all of its subtypes must implement the marker.

Because subtyping is exclusively tied to lifetimes in Rust, and most
marker traits are orthogonal to lifetimes, it actually rarely makes a
difference what choice you make here. But imagine that we have a
marker trait that requires `'static` (such as `Send` today, though
this may change). If we made marker traits covariant with respect to
`Self`, then `&'static Foo : Send` could be used as evidence that `&'x
Foo : Send` for any `'x`, because `&'static Foo <: &'x Foo`:

    (&'static Foo : Send) <: (&'x Foo : Send) // if things were covariant...
    &'static Foo <: &'x Foo                   // ...we'd have the wrong relation here

*Interesting side story: the author thought that covariance would be
correct for some time. It was only when attempting to phrase the
desired behavior as a fn that I realized I had it backward, and
quickly found the counterexample I give above. This gives me
confidence that expressing variance in terms of data and fns is more
reliable than trying to divine the correct results directly.*

# Detailed design

Most of the detailed design has already been covered in the motivation
section.

#### Summary of changes required

- Use variance results to inform subtyping of nominal types
  (structs, enums).
- Use variance for the output type parameters on traits.
- Input type parameters of traits are considered invariant.
- Variance has no effect on the type parameters on an impl or fn;
  rather those are freshly instantiated at each use.
- Report an error if the inference does not find any use of a type or
  lifetime parameter *and* that parameter is not bound in an
  associated type binding in some where clause.

These changes have largely been implemented. You can view the results,
and the impact on the standard library, in
[this branch on nikomatsakis's repository][b]. Note though that as of
the time of this writing, the code is slightly outdated with respect
to this RFC in certain respects (which will clearly be rectified
ASAP).

[b]: https://github.com/nikomatsakis/rust/tree/variance-3

#### Variance inference algorithm

I won't dive too deeply into the inference algorithm that we are using
here. It is based on Section 4 of the paper
["Taming the Wildcards: Combining Definition- and Use-Site Variance"][taming]
published in PLDI'11 and written by Altidor et al. There is a fairly
detailed (and hopefully only slightly outdated) description in
[the code] as well.

[taming]: http://people.cs.umass.edu/~yannis/variance-pldi11.pdf
[the code]: https://github.com/nikomatsakis/rust/blob/variance-3/src/librustc_typeck/variance.rs#L11-L205

#### Bivariance yields an error

One big change from today is that if we compute a result of bivariance
as the variance for any type or lifetime parameter, we will report a
hard error. The error message explicitly suggests the use of a
`PhantomData` or `PhantomFn` marker as appropriate:

    type parameter `T` is never used; either remove it, or use a
    marker such as `std::kinds::marker::PhantomData`"

The goal is to help users as concretely as possible. The documentation
on the phantom markers should also be helpful in guiding users to make
the right choice (the ability to easily attach documentation to the
marker type was in fact the major factor that led us to adopt marker
types in the first place).

#### Rules for associated types

The only exception is when this type parameter is in fact
an output that is implied by where clauses declared on the type.  As
an example of why this distinction is important, consider the type
`Map` declared here:

```rust
struct Map<A,B,I,F>
where I : Iterator<Item=A>, F : FnMut(A) -> B
{
    iter: I,
    fn: F,
}
```

Neither the type `A` nor `B` are reachable from the fields declared
within `Map`, and hence the variance inference for them results in
bivariance. However, they are nonetheless constrained. In the case of
the parameter `A`, its value is determined by the type `I`, and `B` is
determined by the type `F` (note that [RFC 587][587] makes the return
type of `FnMut` an associated type).

The analysis to decide when a type parameter is implied by other type
parameters is the same as that specified in [RFC 447][447].

[447]: https://github.com/rust-lang/rfcs/blob/master/text/0447-no-unused-impl-parameters.md#detailed-design
[587]: https://github.com/rust-lang/rfcs/blob/master/text/0587-fn-return-should-be-an-associated-type.md

# Future possibilities

**Make phantom data and fns more first-class.** One thing I would
consider in the future is to integrate phantom data and fns more
deeply into the language to improve usability. The idea would be to
add a phantom keyword and then permit the explicit declaration of
phantom fields and fns in structs and traits respectively:

```rust
// Instead of
struct Foo<T> {
    pointer: *mut u8,
    _marker: PhantomData<T>
}
trait MarkerTrait : PhantomFn(Self) {
}

// you would write:
struct Foo<T> {
    pointer: *mut u8,
    phantom T
}
trait MarkerTrait {
    phantom fn(Self);
}
```

Phantom fields would not need to be specified when creating an
instance of a type and (being anonymous) could never be named. They
exist solely to aid the analysis. This would improve the usability of
phantom markers greatly.

# Alternatives

**Default to a particular variance when a type or lifetime parameter
is unused.** A prior RFC advocated for this approach, mostly because
markers were seen as annoying to use. However, after some discussion,
it seems that it is more prudent to make a smaller change and retain
explicit declarations. Some factors that influenced this decision:

- The importance of phantom data for other analyses like OIBIT.
- Many unused lifetime parameters (and some unused type parameters) are in
  fact completely unnecessary. Defaulting to a particular variance would
  not help in identifying these cases (though a better dead code lint might).
- There is no default that is always correct but invariance, and
  invariance is typically too strong.
- Phantom type parameters occur relatively rarely anyhow.

**Remove variance inference and use fully explicit declarations.**
Variance inference is a rare case where we do non-local inference
across type declarations. It might seem more consistent to use
explicit declarations. However, variance declarations are notoriously
hard for people to understand. We were unable to come up with a
suitable set of keywords or other system that felt sufficiently
lightweight. Moreover, explicit annotations are error-prone when
compared to the phantom data and fn approach (see example in the
section regarding marker traits).

# Unresolved questions

There is one significant unresolved question: the correct way to
handle a `*mut` pointer. It was revealed recently that while the
current treatment of `*mut T` is correct, it frequently yields overly
conservative inference results in practice. At present the inference
treats `*mut T` as invariant with respect to `T`: this is correct and
sound, because a `*mut` represents aliasable, mutable data, and indeed
the subtyping relation for `*mut T` is that `*mut T <: *mut U if T=U`.

However, in practice, `*mut` pointers are often used to build safe
abstractions, the APIs of which do not in fact permit aliased
mutation. Examples are `Vec`, `Rc`, `HashMap`, and so forth. In all of
these cases, the correct variance is covariant -- but because of the
conservative treatment of `*mut`, all of these types are being
inferred to an invariant result.

The complete solution to this seems to have two parts. First, for
convenience and abstraction, we should not be building safe
abstractions on raw `*mut` pointers anyway. We should have several
convenient newtypes in the standard library, like `ptr::Unique`, that
can be used, which would also help for handling OIBIT conditions and
`NonZero` optimizations. In my branch I have used the existing (but
unstable) type `ptr::Unique` for the primary role, which is kind of an
"unsafe box". `Unique` should ensure that it is covariant with respect
to its argument.

However, this raises the question of how to implement `Unique` under
the hood, and what to do with `*mut T` in general. There are various
options:

1. Change `*mut` so that it behaves like `*const`. This unfortunately
   means that abstractions that introduce shared mutability have
   a responsibility for add phantom data to that affect, something
   like `PhantomData<*const Cell<T>>`. This seems non-obvious and
   unnatural.

2. Rewrite safe abstractions to use `*const` (or even `usize`) instead
   of `*mut`, casting to `*mut` only they have a `&mut self`
   method. This is probably the most conservative option.

3. Change variance to ignore `*mut` referents entirely. Add a lint to
   detect types with a `*mut T` type and require some sort of explicit
   marker that covers `T`. This is perhaps the most explicit
   option. Like option 1, it creates the odd scenario that the
   variance computation and subtyping relation diverge.

Currently I lean towards option 2.
