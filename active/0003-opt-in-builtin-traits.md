- Start Date: 2014-03-24
- RFC PR #: 19
- Rust Issue #: TBD

# Summary

- Rather than determining membership in the builtin traits
  automatically, use `impl` (and `#\[deriving]`) declarations as with
  other traits.
- The compiler will check that for each such `impl` declaration the
  type meets certain criteria (i.e., to implement `Send` for a struct
  `S`, all fields of `S` must have types which are `Send`).
- To check for membership in a builtin trait, we employ a slightly
  modified version of the standard trait matching algorithm.
  Modifications are needed because the language cannot express the
  full set of impls we would require.
- Rename `Pod` trait to `Copy`.

# Motivation

In today's Rust, there are a number of builtin traits (sometimes
called "kinds"): `Send`, `Share`, and `Pod` (in the future, perhaps
`Sized`, but the details of that differ and will addressed in the DST
RFC). These are expressed as traits, but they are quite unlike other
traits in certain ways. One way is that they do not have any methods;
instead, implementing a trait like `Send` indicates that the type has
certain properties (defined below). The biggest difference, though, is
that these traits are not implemented manually by users. Instead, the
compiler decides automatically whether or not a type implements them
based on the contents of the type.

This RFC argues to change this system and instead have users manually
implement the builtin traits for new types that they define.
Naturally there would be `#[deriving]` options as well for
convenience. The compiler's rules (e.g., that a sendable value cannot
reach a non-sendable value) would still be enforced, but at the point
where a builtin trait is explicitly implemented, rather than being
automatically deduced.

There are a couple of reasons to make this change:

1. **Consistency.** All other traits are opt-in, including very common
   traits like `Eq` and `Clone`. It is somewhat surprising that the
   builtin traits act differently.
2. **API Stability.** The builtin traits that are implemented by a
   type are really part of its public API, but unlike other similar
   things they are not declared. This means that seemingly innocent
   changes to the definition of a type can easily break downstream
   users. For example, imagine a type that changes from POD to non-POD
   -- suddenly, all references to instances of that type go from
   copies to moves. Similarly, a type that goes from sendable to
   non-sendable can no longer be used as a message.  By opting in to
   being POD (or sendable, etc), library authors make explicit what
   properties they expect to maintain, and which they do not.
3. **Pedagogy.** Many users find the distinction between pod types
   (which copy) and linear types (which move) to be surprising. Making
   pod-ness opt-in would help to ease this confusion.
4. **Safety and correctness.** In the presence of unsafe code,
   compiler inference is unsound, and it is unfortunate that users
   must remember to "opt out" from inapplicable kinds. There are also
   concerns about future compatibility. Even in safe code, it can also
   be useful to impose additional usage constriants beyond those
   strictly required for type soundness.

More details about these points are provided after the
`Detailed design` section.

# Detailed design

I will first cover the existing builtin traits and define what they
are used for. I will then explain each of the above reasons in more
detail.  Finally, I'll give some syntax examples.

## The builtin traits

We currently define the following builtin traits:

- `Send` -- a type that deeply owns all its contents.
  (Examples: `int`, `~int`, `Cell<int>`, not `&int` or `Rc<int>`)
- `Pod` -- "plain old data" which can be safely copied via memcpy.
  (Examples: `int`, `&int`, not `~int` or `&mut int`)
- `Share` -- a type which is threadsafe when accessed via an `&T`
  reference. (Examples: `int`, `~int`, `&int`, `&mut int`,
  `Atomic<int>`, not `Cell<int>` or `Rc<int>`)

Note that `Pod` is a proper subset of `Send`, but `Send` and `Share`
are unrelated:

- `Cell<uint>` is `Send` but not `Share`.
- `&uint` is `Share` but not `Send`.

## Proposed syntax

Under this proposal, for a struct or enum to be considered send,
share, or pod, those traits must be explicitly implemented:

    struct Foo { ... }
    impl Send for Foo { }
    impl Pod for Foo { }
    impl Share for Foo { }

As usual, deriving forms would be available.

Builtin traits can only be implemented for struct or enum types and
only within the crate in which that struct or enum is defined (see the
section on *Matching and Coherence* below). Whenever a builtin trait
is implemented, the compiler will enforce that all fields or that
struct/enum are of a type which implements the trait (or else of
`Unsafe` type, which matches all traits, see *Matching and
Coherence*).

    struct Foo<'a> { x: &'a int }

    // ERROR: Cannot implement `Send` because the field `x` has type
    // `&'a int` which is not sendable.
    impl<'a> Send for Foo<'a> { }

For generic types, conditional impls are often required to avoid
errors. In the case of `Option<T>`, for example, we must know that the
type `T` implements (e.g.) `Send` before we can implement `Send` for
`Option<T>`:

    enum Option<T> { Some(T), None }
    impl<T> Send for Option<T> { }      // ERROR: T may not implement `Send`

Rewriting that code using a conditional impl would be fine:

    enum Option<T> { Some(T), None }
    impl<T:Send> Send for Option<T> { }      // ERROR: T may not implement `Send`

(This is of course precisely what `#[deriving(Send)]` would generate.)

## Naming of Pod

Part of the proposal is to rename `Pod` to `Copy` so as to better
align the names of the builtin traits (they would not all be verbs).

## Copy and linearity

One of the most important aspects of this proposal is that the `Copy`
trait would be something that one "opts in" to. This means that
structs and enums would *move by default* unless their type is
explicitly declared to be `Copy`. So, for example, the following code
would be in error:

    struct Point { x: int, y: int }
    ...
    let p = Point { x: 1, y: 2 };
    let q = p;  // moves p
    print(p.x); // ERROR

To allow that example, one would have to impl `Copy` for `Point`:

    struct Point { x: int, y: int }
    impl Copy for Point { }
    ...
    let p = Point { x: 1, y: 2 };
    let q = p;  // copies p, because Point is Pod
    print(p.x); // OK

Effectively this change introduces a three step ladder for types:

1. If you do nothing, your type is *linear*, meaning that it moves
   from place to place and can never be copied in any way. (We need a
   better name for that.)
2. If you implement `Clone`, your type is *cloneable*, meaning that it
   moves from place to place, but it can be explicitly cloned. This is
   suitable for cases where copying is expensive.
3. If you implement `Copy`, your type is *copyable*, meaning that
   it is just copied by default without the need for an explicit
   clone.  This is suitable for small bits of data like ints or
   points.

What is nice about this change is that when a type is defined, the
user makes an *explicit choice* between these three options.

## Matching and coherence

In general, determining whether a type implements a builtin trait can
follow the existing trait matching algorithm, but it will have to be
somewhat specialized. The problem is that we are somewhat limited in
the kinds of impls that we can write, so some of the implementations
we would want must be "hard-coded".

Specifically we are limited around tuples, fixed-length array types,
proc types, closure types, and trait types:

- *Fixed-length arrays:* A fixed-length array `[T, ..n]` is `Send/Copy/Share`
  if `T` is `Send/Copy/Share`, regardless of `n`. (Conceivably, we could
  also say that if `n` is `0`, then `[T, ..n]` is `Send/Copy/Share` regardless
  of `T`).
- *Tuples*: A tuple `(T_0, ..., T_n)` is `Send/Copy/Share` depending
  if, for all `i`, `T_i` is `Send/Copy/Share`.
- *Closures*: A closure type `|T_0, ..., T_n|:K -> T_n+1` is never
  `Send` nor `Copy`. It is `Share` iff `K` is `Share`.
- *Procs*: A proc type `proc(T_0, ..., T_n):K -> T_n+1` is
  never `Copy`. It is `Send/Share` iff `K` is `Send/Share`.
- *Trait objects*: A trait object type `Trait:K` (assuming DST here ;) is
  never `Copy`. It may be `Send/Share` iff `K` is `Send/Share`.

We cannot currently express the above conditions using impls. We may
at some point in the future grow the ability to express some of them.
For now, though, these "impls" will be hardcoded into the algorithm.

Otherwise, the complete list of builtin impls is roughly like this
(undoubtedly I am missing a few things):

    trait Send;
    trait Share;
    trait Copy; // aka Pod

    impl Copy for "scalars like uint, u8, etc" { }
    impl<T> Copy for *T { }
    impl<'a,T> Copy for &'a T { }

    impl Send for "scalars like uint, u8, etc" { }
    impl<T:Send> for *T { }
    impl<T:Send> for ~T { }

    impl Share for "scalars like uint, u8, etc" { }
    impl<T:Share> for *T { }
    impl<T:Share> for ~T { }
    impl<'a,T:Share> for &'a T { }
    impl<'a,T:Share> for &'a mut T { } // (if this surprises you, see * below)

Per the usual coherence rules, since we will have the above impls in
`libstd`, and we will have impls for types like tuples and
fixed-length arrays baked in, the only impls that end users are
permitted to write are impls for struct and enum types that they
define themselves. This is simply an extra coherence rule, hard-coded
because some of the impls (e.g., for tuples) are hard-coded.

(\*) Wait, `&mut T` is `Share`? How is that threadsafe?

Somewhat surprisingly, `&mut T` is share. Remember, a type `U` is
share if all possible operations on `&U` are threadsafe. In this case,
`U` is `&mut T`, this means we have to consider what operations are
possible on a `& &mut T`. In that case ,the `&mut T` is found in an
aliasable location and hence is immutable (if you can find a counter
example, that's definitely a bug).

Moreover, there is one further exception to the rules.  The
`Unsafe<T>` type is *always* considered to implement `Share`, no
matter the type `T`. `Send` and `Copy` are implemented if `T` is
`Send` and `Copy`. The motivation here is that we want to be able to
permit a type like `Mutex` to be `Share` even if it closes over data
that is not `Share`.

# Implementation plan

Here is a loose implementation plan that @flaper87 and I worked
out. No doubt things will change along the way.

1. Create a nicely encapsulated subroutine S to check whether type T
   meets bound B For example, to test that some type T is Pod. @eddyb
   did something recently you can use as an example, where he added
   some code to do vtable matching for the Drop trait from trans.  One
   catch is that we will definitely want some sort of cache.

2. Modify the vtable code to handle builtin bounds and add builtin
   impls (see below)
   - We'll need special code to accommodate the types detailed above

3. Use the subroutine S in moves.rs to do the "is pod" check.

4. Same for rustc::middle::kind, except that we should move the "check
   bounds on type parameters" into type check.
   - Why do this? Because these checks will now be so close to vtable
     matching it no longer makes sense to do them in `kind.rs`

5. Check to make sure that the impls the user provides are safe:
   - User-defined impls can only apply to enums or structs
    - If implementing a builtin trait T for a struct type S, each
      field of S must have a type that implements S.
    - same for enums, but "for each variant, for each argument" essentially

# Expanded motivation

Now that the detailed design is presented, I wanted to expand more on
the motivation.

## Consistency

This change would bring the builtin traits more in line with other
common traits, such as `Eq` and `Clone`. On a historical note, this
proposal continues a trend, in that both of those operations used to
be natively implemented by the compiler as well.

## API Stability

The set of builtin traits implemented by a type must be considered
part of its public inferface. At present, though, it's quite invisible
and not under user control. If a type is changed from `Pod` to
non-pod, or `Send` to non-send, no error message will result until
client code attempts to use an instance of that type. In general we
have tried to avoid this sort of situation, and instead have each
declaration contain enough information to check it indepenently of its
uses. Issue #12202 describes this same concern, specifically with
respect to stability attributes.

Making opt-in explicit effectively solves this problem. It is clearly
written out which traits a type is expected to fulfill, and if the
type is changed in such a way as to violate one of these traits, an
error will be reported at the `impl` site (or `#[deriving]`
declaration).

## Pedagogy

When users first start with Rust, ownership and ownership transfer is
one of the first things that they must learn. This is made more
confusing by the fact that types are automatically divided into pod
and non-pod without any sort of declaration. It is not necessarily
obvious why a `T` and `~T` value, which are *semantically equivalent*,
behave so differently by default. Makes the pod category something you
opt into means that types will all be linear by default, which can
make teaching and leaning easier.

## Safety and correctness: unsafe code

For safe code, the compiler's rules for deciding whether or not a type
is sendable (and so forth) are perfectly sound. However, when unsafe
code is involved, the compiler may draw the wrong conclusion. For such
cases, types must *opt out* of the builtin traits.

In general, the *opt out* approach seems to be hard to reason about:
many people (including myself) find it easier to think about what
properties a type *has* than what properties it *does not* have,
though clearly the two are logically equivalent in this binary world
we programmer's inhabit.

More concretely, opt out is dangerous because it means that types with
unsafe methods are generally *wrong by default*. As an example,
consider the definition of the `Cell` type:

    struct Cell<T> {
        priv value: T
    }

This is a perfectly ordinary struct, and hence the compiler would
conclude that cells are freezable (if `T` is freezable) and so forth.
However, the *methods* attached to `Cell` use unsafe magic to mutate
`value`, even when the `Cell` is aliased:

    impl<T:Pod> Cell<T> {
        pub fn set(&self, value: T) {
            unsafe {
                *cast::transmute_mut(&self.value) = value
            }
        }
    }

To accommodate this, we currently use *marker types* -- special types
known to the compiler which are considered nonpod and so forth. Therefore,
the full definition of `Cell` is in fact:

    pub struct Cell<T> {
        priv value: T,
        priv marker1: marker::InvariantType<T>,
        priv marker2: marker::NoFreeze,
    }

Note the two markers. The first, `marker1`, is a hint to the variance
engine indicating that the type `Cell` must be invariant with respect
to its type argument. The second, `marker2`, indicates that `Cell` is
non-freeze. This then informs the compiler that the referent of a
`&Cell<T>` can't be considered immutable. The problem here is that, if
you don't know to opt-out, you'll wind up with a type definition that
is unsafe.

This argument is rather weakened by the continued necessity of a
`marker::InvariantType` marker. This could be read as an argument
towards explicit variance. However, I think that in this particular
case, the better solution is to introduce the `Mut<T>` type described
in #12577 -- the `Mut<T>` type would give us the invariance.

Using `Mut<T>` brings us back to a world where any type that uses
`Mut<T>` to obtain interior mutability is correct by default, at least
with respect to the builtin kinds. Types like `Atomic<T>` and
`Volatile<T>`, which guarantee data race freedom, would therefore have
to *opt in* to the `Share` kind, and types like `Cell<T>` would simply
do nothing.

## Safety and correctness: future compatibility

Another concern about having the compiler automatically infer
membership into builtin bounds is that we may find cause to add new
bounds in the future. In that case, existing Rust code which uses
unsafe methods might be inferred incorrectly, because it would not
know to opt out of those future bounds. Therefore, any future bounds
will *have* to be opt out anyway, so perhaps it is best to be
consistent from the start.

## Safety and correctness: semantic constraints

Even if type safety is maintained, some types ought not to be copied
for semantic reasons. An example from the compiler is the
`Datum<Rvalue>` type, which is used in code generation to represent
the computed result of an rvalue expression. At present, the type
`Rvalue` implements a (empty) destructor -- the sole purpose of this
destructor is to ensure that datums are not consumed more than once,
because this would likely correspond to a code gen bug, as it would
mean that the result of the expression evaluation is consumed more
than once. Another example might be a newtype'd integer used for
indexing into a thread-local array: such a value ought not to be
sendable. And so forth. Using marker types for these kinds of
situations, or empty destructors, is very awkward. Under this
proposal, users needs merely refrain from implementing the relevant
traits.

# Alternatives and counterarguments

The downsides of this proposal are:

- There is some annotation burden. I had intended to gather statistics
  to try and measure this but have not had the time.

- If a library forgets to implement all the relevant traits for a
  type, there is little recourse for users of that library beyond pull
  requests to the original repository. This is already true with
  traits like `Eq` and `Ord`. However, as SiegeLord noted on IRC, that
  you can often work around the absence of `Eq` with a newtype
  wrapper, but this is not true if a type fails to implement `Send` or
  `Copy`. This danger (forgetting to implement traits) is essentially
  the counterbalance to the "forward compatbility" case made above:
  where implementing traits by default means types may implement too
  much, forcing explicit opt in means types may implement too little.
  One way to mitigate this problem would be to have a lint for when an
  impl of some kind (etc) would be legal, but isn't implemented, at
  least for publicly exported types in library crates.

What other designs have been considered? What is the impact of not doing this?

# Unresolved questions

Do we want some kind of shorthand for common trait combinations?  I
originally proposed `Data` but we couldn't settle on what a useful set
of trait combinations would be. This can easily be added later.
