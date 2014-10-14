- Start Date: 2014-09-18
- RFC PR #: [rust-lang/rfcs#19](https://github.com/rust-lang/rfcs/pull/19), [rust-lang/rfcs#127](https://github.com/rust-lang/rfcs/pull/127)
- Rust Issue #: [rust-lang/rust#13231](https://github.com/rust-lang/rust/issues/13231)

# Summary

The high-level idea is to add language features that simultaneously
achieve three goals:

1. move `Send` and `Share` out of the language entirely and into the
   standard library, providing mechanisms for end users to easily
   implement and use similar "marker" traits of their own devising;
2. make "normal" Rust types sendable and sharable by default, without
   the need for explicit opt-in; and,
3. continue to require "unsafe" Rust types (those that manipulate
   unsafe pointers or implement special abstractions) to "opt-in" to
   sendability and sharability with an unsafe declaration.
   
These goals are achieved by two changes:

1. **Unsafe traits:** An *unsafe trait* is a trait that is unsafe to
   implement, because it represents some kind of trusted
   assertion. Note that unsafe traits are perfectly safe to
   *use*. `Send` and `Share` are examples of unsafe traits:
   implementing these traits is effectively an assertion that your
   type is safe for threading.
2. **Default and negative impls:** A *default impl* is one that
   applies to all types, except for those types that explicitly *opt
   out*. For example, there would be a default impl for `Send`,
   indicating that all types are `Send` "by default".
   
   To counteract a default impl, one uses a *negative impl* that
   explicitly opts out for a given type `T` and any type that contains
   `T`. For example, this RFC proposes that unsafe pointers `*T` will
   opt out of `Send` and `Share`. This implies that unsafe pointers
   cannot be sent or shared between threads by default. It also
   implies that any structs which contain an unsafe pointer cannot be
   sent. In all examples encountered thus far, the set of negative
   impls is fixed and can easily be declared along with the trait
   itself.
   
   Safe wrappers like `Arc`, `Atomic`, or `Mutex` can opt to implement
   `Send` and `Share` explicitly. This will then make them be
   considered sendable (or sharable) even though they contain unsafe
   pointers etc.
   
Based on these two mechanisms, we can remove the notion of `Send` and
`Share` as builtin concepts. Instead, these would become unsafe traits
with default impls (defined purely in the library). The library would
explicitly *opt out* of `Send`/`Share` for certain types, like unsafe
pointers (`*T`) or interior mutability (`Unsafe<T>`). Any type,
therefore, which contains an unsafe pointer would be confined (by
default) to a single thread. Safe wrappers around those types, like
`Arc`, `Atomic`, or `Mutex`, can then opt back in by explicitly
implementing `Send` (these impls would have to be designed as unsafe).

# Motivation

Since proposing opt-in builtin traits, I have become increasingly
concerned about the notion of having `Send` and `Share` be strictly
opt-in. There are two main reasons for my concern:

1. Rust is very close to being a language where computations can be
   parallelized by default. Making `Send`, and *especially* `Share`,
   opt-in makes that harder to achieve.
2. The model followed by `Send`/`Share` cannot easily be extended to
   other traits in the future nor can it be extended by end-users with
   their own similar traits. It is worrisome that I have come across
   several use cases already which might require such extension
   (described below).   

To elaborate on those two points: With respect to parallelization: for
the most part, Rust types are threadsafe "by default". To make
something non-threadsafe, you must employ unsychronized interior
mutability (e.g., `Cell`, `RefCell`) or unsychronized shared ownership
(`Rc`). In both cases, there are also synchronized variants available
(`Mutex`, `Arc`, etc). This implies that we can make APIs to enable
intra-task parallelism and they will work ubiquitously, so long as
people avoid `Cell` and `Rc` when not needed. Explicit opt-in
threatens that future, however, because fewer types will implement
`Share`, even if they are in fact threadsafe.
   
With respect to extensibility, it is partiularly worrisome that if a
library forgets to implement `Send` or `Share`, downstream clients are
stuck. They cannot, for example, use a newtype wrapper, because it
would be illegal to implement `Send` on the newtype. This implies that
all libraries must be vigilant about implementing `Send` and `Share`
(even more so than with other pervasive traits like `Eq` or `Ord`).
The current plan is to address this via lints and perhaps some
convenient deriving syntax, which may be adequate for `Send` and
`Share`. But if we wish to add new "classification" traits in the
future, these new traits won't have been around from the start, and
hence won't be implemented by all existing code.

Another concern of mine is that end users cannot define classification
traits of their own. For example, one might like to define a trait for
"tainted" data, and then test to ensure that tainted data doesn't pass
through some generic routine. There is no particular way to do this
today.

More examples of classification traits that have come up recently in
various discussions:

- `Snapshot` (nee `Freeze`), which defines *logical* immutability
  rather than *physical* immutability. `Rc<int>`, for example, would
  be considered `Snapshot`. `Snapshot` could be useful because
  `Snapshot+Clone` indicates a type whose value can be safely
  "preserved" by cloning it.
- `NoManaged`, a type which does not contain managed data. This might
  be useful for integrating garbage collection with custom allocators
  which do not wish to serve as potential roots.
- `NoDrop`, a type which does not contain an explicit destructor. This
  can be used to avoid nasty GC quandries.

All three of these (`Snapshot`, `NoManaged`, `NoDrop`) can be easily
defined using traits with default impls.

A final, somewhat weaker, motivator is aesthetics. Ownership has allowed
us to move threading almost entirely into libaries. The one exception
is that the `Send` and `Share` types remain built-in. Opt-in traits
makes them *less* built-in, but still requires custom logic in the
"impl matching" code as well as special safety checks when
`Safe` or `Share` are implemented.

After the changes I propose, the only traits which would be
specicially understood by the compiler are `Copy` and `Sized`. I
consider this acceptable, since those two traits are intimately tied
to the core Rust type system, unlike `Send` and `Share`.

# Detailed design

## Unsafe traits

Certain traits like `Send` and `Share` are critical to memory safety.
Nonetheless, it is not feasible to check the thread-safety of all
types that implement `Send` and `Share`. Therefore, we introduce a
notion of an *unsafe trait* -- this is a trait that is unsafe to
implement, because implementing it carries semantic guarantees that,
if compromised, threaten memory safety in a deep way.

An unsafe trait is declared like so:

    unsafe trait Foo { ... }
    
To implement an unsafe trait, one must mark the impl as unsafe:

    unsafe impl Foo for Bar { ... }
    
Designating an impl as unsafe does not automatically mean that the
body of the methods is an unsafe block. Each method in the trait must
also be declared as unsafe if it to be considered unsafe.

Unsafe traits are only unsafe to *implement*. It is always safe to
reference an unsafe trait. For example, the following function is
safe:

    fn foo<T:Send>(x: T) { ... }
    
It is also safe to *opt out* of an unsafe trait (as discussed in the
next section).
    
## Default and negative impls

We add a notion of a *default impl*, written:

    impl Trait for .. { }
    
Default impls are subject to various limitations:

1. The default impl must appear in the same module as `Trait` (or a submodule).
2. `Trait` must not define any methods.

We further add the notion of a *negative impl*, written:

    impl !Trait for Foo { }
    
Negative impls are only permitted if `Trait` has a default impl.
Negative impls are subject to the usual orphan rules, but they are
permitting to be overlapping. This makes sense because negative impls
are not providing an implementation and hence we are not forced to
select between them. For similar reasons, negative impls never need to
be marked unsafe, even if they reference an unsafe trait.

Intuitively, to check whether a trait `Foo` that contains a default
impl is implemented for some type `T`, we first check for explicit
(positive) impls that apply to `T`. If any are found, then `T`
implements `Foo`. Otherwise, we check for negative impls. If any are
found, then `T` does not implement `Foo`. If neither positive nor
negative impls were found, we proceed to check the component types of
`T` (i.e., the types of a struct's fields) to determine whether all of
them implement `Foo`. If so, then `Foo` is considered implemented by
`T`.

Oe non-obvious part of the procedure is that, as we recursively
examine the component types of `T`, we add to our list of assumptions
that `T` implements `Foo`. This allows recursive types like

    struct List<T> { data: T, next: Option<List<T>> }

to be checked successfully. Otherwise, we would recursive infinitely.
(This procedure is directly analagous to what the existing
`TypeContents` code does.)

Note that there exist types that expand to an infinite tree of types.
Such types cannot be successfully checked with a recursive impl; they
will simply overflow the builtin depth checking. However, such types
also break code generation under monomorphization (we cannot create a
finite set of LLVM types that correspond to them) and are in general
not supported. Here is an example of such a type:

    struct Foo<A> {
        data: Option<Foo<Vec<A>>>
    }

The difference between `Foo` and `List` above is that `Foo<A>`
references `Foo<Vec<A>>`, which will then in turn reference
`Foo<Vec<Vec<A>>>` and so on.

## Modeling Send and Share using default traits

The `Send` and `Share` traits will be modeled entirely in the library
as follows. First, we declare the two traits as follows:

    unsafe trait Send { }
    unsafe impl Send for .. { }
    
    unsafe trait Share { }
    unsafe impl Share for .. { }
    
Both traits are declared as unsafe because declaring that a type if
`Send` and `Share` has ramifications for memory safety (and data-race
freedom) that the compiler cannot, itself, check.

Next, we will add *opt out* impls of `Send` and `Share` for the
various unsafe types:

    impl<T> !Send for *T { }
    impl<T> !Share for *T { }

    impl<T> !Send for *mut T { }
    impl<T> !Share for *mut T { }

    impl<T> !Share for Unsafe<T> { }
    
Note that it is not necessary to write unsafe to *opt out* of an
unsafe trait, as that is the default state.

Finally, we will add *opt in* impls of `Send` and `Share` for the
various safe wrapper types as needed. Here I give one example, which
is `Mutex`. `Mutex` is interesting because it has the property that it
converts a type `T` from being `Sendable` to something `Sharable`:

    unsafe impl<T:Send> Send for Mutex<T> { }
    unsafe impl<T:Send> Share for Mutex<T> { }

## The `Copy` and `Sized` traits

The final two builtin traits are `Copy` and `Share`. This RFC does not
propose any changes to those two traits but rather relies on the
specification from [the original opt-in RFC](0003-opt-in-builtin-traits.md).

### Controlling copy vs move with the `Copy` trait

The `Copy` trait is "opt-in" for user-declared structs and enums. A
struct or enum type is considered to implement the `Copy` trait only
if it implements the `Copy` trait.  This means that structs and enums
would *move by default* unless their type is explicitly declared to be
`Copy`. So, for example, the following code would be in error:

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

Effectively, there is a three step ladder for types:

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

### Determining whether a type is `Sized`

Per the DST specification, the array types `[T]` and object types like
`Trait` are unsized, as are any structs that embed one of those
types. The `Sized` trait can never be explicitly implemented and
membership in the trait is always automatically determined.

### Matching and coherence for the builtin types `Copy` and `Sized`

In general, determining whether a type implements a builtin trait can
follow the existing trait matching algorithm, but it will have to be
somewhat specialized. The problem is that we are somewhat limited in
the kinds of impls that we can write, so some of the implementations
we would want must be "hard-coded".

Specifically we are limited around tuples, fixed-length array types,
proc types, closure types, and trait types:

- *Fixed-length arrays:* A fixed-length array `[T, ..n]` is `Copy`
  if `T` is `Copy`. It is always `Sized` as `T` is required to be `Sized`.
- *Tuples*: A tuple `(T_0, ..., T_n)` is `Copy/Sized` depending if,
  for all `i`, `T_i` is `Copy/Sized`.
- *Trait objects* (including procs and closures): A trait object type
  `Trait:K` (assuming DST here ;) is never `Copy` nor `Sized`.

We cannot currently express the above conditions using impls. We may
at some point in the future grow the ability to express some of them.
For now, though, these "impls" will be hardcoded into the algorithm as
if they were written in libstd.

Per the usual coherence rules, since we will have the above impls in
`libstd`, and we will have impls for types like tuples and
fixed-length arrays baked in, the only impls that end users are
permitted to write are impls for struct and enum types that they
define themselves. Although this rule is in the general spirit of the
coherence checks, it will have to be written specially.

# Design discussion

#### Why unsafe traits

Without unsafe traits, it would be possible to
create data races without using the `unsafe` keyword:

    struct MyStruct { foo: Cell<int> }
    impl Share for MyStruct { }

#### Balancing abstraction, safety, and convenience.

In general, the existence of default traits is *anti-abstraction*, in
the sense that it exposes implementation details a library might
prefer to hide. Specifically, adding new private fields can cause your
types to become non-sendable or non-sharable, which may break
downstream clients without your knowing. This is a known challenge
with parallelism: knowing whether it is safe to parallelize relies on
implementation details we have traditionally tried to keep secret from
clients (often it is said that parallelism is "anti-modular" or
"anti-compositional" for this reason).

I think this risk must be weighed against the limitations of requiring
total opt in. Requiring total opt in not only means that some types
will accidentally fail to implement send or share when they could, but
it also means that libraries which wish to employ marker traits cannot
be composed with other libraries that are not aware of those marker
traits. In effect, opt-in is anti-modular in its own way.

To be more specific, imagine that library A wishes to define a
`Untainted` trait, and it specifically opts out of `Untainted` for
some base set of types. It then wishes to have routines that only
operate on `Untained` data. Now imagine that there is some other
library B that defines a nifty replacement for `Vector`,
`NiftyVector`. Finally, some library C wishes to use a
`NiftyVector<uint>`, which should not be considered tainted, because
it doesn't reference any tainted strings. However, `NiftyVector<uint>`
does not implement `Untainted` (nor can it, without either library A
or libary B knowing about one another). Similar problems arise for any
trait, of course, due to our coherence rules, but often they can be
overcome with new types. Not so with `Send` and `Share`.

#### Other use cases

Part of the design involves making space for other use cases. I'd like
to skech out how some of those use cases can be implemented briefly.
This is not included in the *Detailed design* section of the RFC
because these traits generally concern other features and would be
added under RFCs of their own.

**Isolating snapshot types.** It is useful to be able to identify
types which, when cloned, result in a logical *snapshot*. That is, a
value which can never be mutated.  Note that there may in fact be
mutation under the covers, but this mutation is not visible to the
user. An example of such a type is `Rc<T>` -- although the ref count
on the `Rc` may change, the user has no direct access and so `Rc<T>`
is still logically snapshotable.  However, not all `Rc` instances are
snapshottable -- in particular, something like `Rc<Cell<int>>` is not.

    trait Snapshot { }
    impl Snapshot for .. { }
    
    // In general, anything that can reach interior mutability is not
    // snapshotable.
    impl<T> !Snapshot for Unsafe<T> { }
    
    // But it's ok for Rc<T>.
    impl<T:Snapshot> Snapshot for Rc<T> { }

Note that these definitions could all occur in a library. That is, the
`Rc` type itself doesn't need to know about the `Snapshot` trait.

**Preventing access to managed data.** As part of the GC design, we
expect it will be useful to write specialized allocators or smart
pointers that explicitly do *not* support tracing, so as to avoid any
kind of GC overhead. The general idea is that there should be a bound,
let's call it `NoManaged`, that indicates that a type cannot reach
managed data and hence does not need to be part of the GC's root
set. This trait could be implemented as follows:

    unsafe trait NoManaged { }
    unsafe impl NoManaged for .. { }
    impl<T> !NoManaged for Gc<T> { }

**Preventing access to destructors.** It is generally recognized that
allowing destructors to escape into managed data -- frequently
referred to as finalizers -- is a bad idea.  Therefore, we would
generally like to ensure that anything is placed into a managed box
does not implement the drop trait. Instead, we would prefer to regular
the use of drop through a guardian-like API, which basically means
that destructors are not asynchronously executed by the GC, as they
would be in Java, but rather enqueued for the mutator thread to run
synchronously at its leisure. In order to handle this, though, we
presumably need some sort of guardian wrapper types that can take a
value which has a destructor and allow it to be embedded within
managed data. We can summarize this in a trait `GcSafe` as follows:

    unsafe trait GcSafe { }
    unsafe impl GcSafe for .. { }

    // By default, anything which has drop trait is not GcSafe.
    impl<T:Drop> !GcSafe for T { }
    
    // But guardians are, even if `T` has drop.
    impl<T> GcSafe for Guardian<T> { }

#### Why are `Copy` and `Sized` different?

The `Copy` and `Sized` traits remain builtin to the compiler. This
makes sense because they are intimately tied to analyses the compiler
performs. For example, the running of destructors and tracking of
moves requires knowing which types are `Copy`. Similarly, the
allocation of stack frames need to know whether types are fully
`Sized`. In contrast, sendability and sharability has been fully
exported to libraries at this point.

In addition, opting in to `Copy` makes sense for several reasons:

- Experience has shown that "data-like structs", for which `Copy` is
  most appropriate, are a very small percentage of the total.
- Changing a public API from being copyable to being only movable has
  a outsized impact on users of the API. It is common however that as
  APIs evolve they will come to require owned data (like a `Vec`),
  even if they do not initially, and hence will change from being
  copyable to only movable. Opting in to `Copy` is a way of saying
  that you never foresee this coming to pass.
- Often it is useful to create linear "tokens" that do not themselves
  have data but represent permissions. This can be done today using
  markers but it is awkward. It becomes much more natural under this
  proposal.

# Drawbacks

**API stability.** The main drawback of this approach over the
existing opt-in approach seems to be that a type may be "accidentally"
sendable or sharable. I discuss this above under the heading of
"balancing abstraction, safety, and convenience". One point I would
like to add here, as it specifically pertains to API stability, is
that a library may, if they choose, opt out of `Send` and `Share`
pre-emptively, in order to "reserve the right" to add non-sendable
things in the future.

# Alternatives

- The existing opt-in design is of course an alternative.

- We could also simply add the notion of `unsafe` traits and *not*
  default impls and then allow types to unsafely implement `Send` or
  `Share`, bypassing the normal safety guidelines. This gives an
  escape valve for a downstream client to assert that something is
  sendable which was not declared as sendable. However, such a
  solution is deeply unsatisfactory, because it rests on the
  downstream client making an assertion about the implementation of
  the library it uses. If that library should be updated, the client's
  assumptions could be invalidated, but no compilation errors will
  result (the impl was already declared as unsafe, after all).

# Phasing

Many of the mechanisms described in this RFC are not needed
immediately.  Therefore, we would like to implement a minimal
"forwards compatible" set of changes now and then leave the remaining
work for after the 1.0 release. The builtin rules that the compiler
currently implements for send and share are quite close to what is
proposed in this RFC. The major change is that unsafe pointers and the
`UnsafeCell` type are currently considered sendable.

Therefore, to be forwards compatible in the short term, we can use the
same hybrid of builtin and explicit impls for `Send` and `Share` that
we use for `Copy`, with the rule that unsafe pointers and `UnsafeCell`
are not considered sendable. We must also implement the `unsafe trait`
and `unsafe impl` concept.

What this means in practice is that using `*const T`, `*mut T`, and
`UnsafeCell` will make a type `T` non-sendable and non-sharable, and
`T` must then explicitly implement `Send` or `Share`.

# Unresolved questions

- The terminology of "unsafe trait" seems somewhat misleading, since
  it seems to suggest that "using" the trait is unsafe, rather than
  implementing it. One suggestion for an alternate keyword was
  `trusted trait`, which might dovetail with the use of `trusted` to
  specify a trusted block of code. If we did use `trusted trait`, it
  seems that all impls would also have to be `trusted impl`.
- Perhaps we should declare a trait as a "default trait" directly,
  rather than using the `impl Drop for ..` syntax. I don't know
  precisely what syntax to use, though.
- Currently, there are special rules relating to object types and
  the builtin traits. If the "builtin" traits are no longer builtin,
  we will have to generalize object types to be simply a set of trait
  references. This is already planned but merits a second RFC. Note
  that no changes here are required for the 1.0, since the phasing
  plan dictates that builtin traits remain special until after 1.0.
