- Start Date: (fill me in with today's date, 2014-08-04)
- RFC PR #: [rust-lang/rfcs#195](https://github.com/rust-lang/rfcs/pull/195)
- Rust Issue #: [rust-lang/rust#17307](https://github.com/rust-lang/rust/issues/17307)

# Summary

This RFC extends traits with *associated items*, which make generic programming
more convenient, scalable, and powerful. In particular, traits will consist of a
set of methods, together with:

* Associated functions (already present as "static" functions)
* Associated consts
* Associated types
* Associated lifetimes

These additions make it much easier to group together a set of related types,
functions, and constants into a single package.

This RFC also provides a mechanism for *multidispatch* traits, where the `impl`
is selected based on multiple types. The connection to associated items will
become clear in the detailed text below.

*Note: This RFC was originally accepted before RFC 246 introduced the
distinction between const and static items. The text has been updated to clarify
that associated consts will be added rather than statics, and to provide a
summary of restrictions on the initial implementation of associated
consts. Other than that modification, the proposal has not been changed to
reflect newer Rust features or syntax.*

# Motivation

A typical example where associated items are helpful is data structures like
graphs, which involve at least three types: nodes, edges, and the graph itself.

In today's Rust, to capture graphs as a generic trait, you have to take the
additional types associated with a graph as _parameters_:

```rust
trait Graph<N, E> {
    fn has_edge(&self, &N, &N) -> bool;
    ...
}
```

The fact that the node and edge types are parameters is confusing, since any
concrete graph type is associated with a *unique* node and edge type. It is also
inconvenient, because code working with generic graphs is likewise forced to
parameterize, even when not all of the types are relevant:

```rust
fn distance<N, E, G: Graph<N, E>>(graph: &G, start: &N, end: &N) -> uint { ... }
```

With associated types, the graph trait can instead make clear that the node and
edge types are determined by any `impl`:

```rust
trait Graph {
    type N;
    type E;
    fn has_edge(&self, &N, &N) -> bool;
}
```

and clients can abstract over them all at once, referring to them through the
graph type:

```rust
fn distance<G: Graph>(graph: &G, start: &G::N, end: &G::N) -> uint { ... }
```

The following subsections expand on the above benefits of associated items, as
well as some others.

## Associated types: engineering benefits for generics

As the graph example above illustrates, associated _types_ do not increase the
expressiveness of traits _per se_, because you can always use extra type
parameters to a trait instead. However, associated types provide several
engineering benefits:

* **Readability and scalability**

  Associated types make it possible to abstract over a whole family of types at
  once, without having to separately name each of them. This improves the
  readability of generic code (like the `distance` function above).  It also
  makes generics more "scalable": traits can incorporate additional associated
  types without imposing an extra burden on clients that don't care about those
  types.

  In today's Rust, by contrast, adding additional generic parameters to a
  trait often feels like a very "heavyweight" move.

* **Ease of refactoring/evolution**

  Because users of a trait do not have to separately parameterize over its
  associated types, new associated types can be added without breaking all
  existing client code.

  In today's Rust, by contrast, associated types can only be added by adding
  more type parameters to a trait, which breaks all code mentioning the trait.

## Clearer trait matching

Type parameters to traits can either be "inputs" or "outputs":

* **Inputs**. An "input" type parameter is used to _determine_ which `impl` to
  use.

* **Outputs**. An "output" type parameter is uniquely determined _by_ the
  `impl`, but plays no role in selecting the `impl`.

Input and output types play an important role for type inference and trait
coherence rules, which is described in more detail later on.

In the vast majority of current libraries, the only input type is the `Self`
type implementing the trait, and all other trait type parameters are outputs.
For example, the trait `Iterator<A>` takes a type parameter `A` for the elements
being iterated over, but this type is always determined by the concrete `Self`
type (e.g. `Items<u8>`) implementing the trait: `A` is typically an output.

Additional input type parameters are useful for cases like binary operators,
where you may want the `impl` to depend on the types of *both*
arguments. For example, you might want a trait

```rust
trait Add<Rhs, Sum> {
    fn add(&self, rhs: &Rhs) -> Sum;
}
```

to view the `Self` and `Rhs` types as inputs, and the `Sum` type as an output
(since it is uniquely determined by the argument types). This would allow
`impl`s to vary depending on the `Rhs` type, even though the `Self` type is the same:

```rust
impl Add<int, int> for int { ... }
impl Add<Complex, Complex> for int { ... }
```

Today's Rust does not make a clear distinction between input and output type
parameters to traits. If you attempted to provide the two `impl`s above, you
would receive an error like:

```
error: conflicting implementations for trait `Add`
```

This RFC clarifies trait matching by:

* Treating all trait type parameters as *input* types, and
* Providing associated types, which are *output* types.

In this design, the `Add` trait would be written and implemented as follows:

```rust
// Self and Rhs are *inputs*
trait Add<Rhs> {
    type Sum; // Sum is an *output*
    fn add(&self, &Rhs) -> Sum;
}

impl Add<int> for int {
    type Sum = int;
    fn add(&self, rhs: &int) -> int { ... }
}

impl Add<Complex> for int {
    type Sum = Complex;
    fn add(&self, rhs: &Complex) -> Complex { ... }
}
```

With this approach, a trait declaration like `trait Add<Rhs> { ... }` is really
defining a *family* of traits, one for each choice of `Rhs`. One can then
provide a distinct `impl` for every member of this family.

## Expressiveness

Associated types, lifetimes, and functions can already be expressed in today's
Rust, though it is unwieldy to do so (as argued above).

But associated _consts_ cannot be expressed using today's traits.

For example, today's Rust includes a variety of numeric traits, including
`Float`, which must currently expose constants as static functions:

```rust
trait Float {
    fn nan() -> Self;
    fn infinity() -> Self;
    fn neg_infinity() -> Self;
    fn neg_zero() -> Self;
    fn pi() -> Self;
    fn two_pi() -> Self;
    ...
}
```

Because these functions cannot be used in constant expressions, the modules for
float types _also_ export a separate set of constants as consts, not using
traits.

Associated constants would allow the consts to live directly on the traits:

```rust
trait Float {
    const NAN: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    const NEG_ZERO: Self;
    const PI: Self;
    const TWO_PI: Self;
    ...
}
```

## Why now?

The above motivations aside, it may not be obvious why adding associated types
*now* (i.e., pre-1.0) is important. There are essentially two reasons.

First, the design presented here is *not* backwards compatible, because it
re-interprets trait type parameters as inputs for the purposes of trait
matching. The input/output distinction has several ramifications on coherence
rules, type inference, and resolution, which are all described later on in the
RFC.

Of course, it might be possible to give a somewhat less ideal design where
associated types can be added later on without changing the interpretation of
existing trait type parameters. For example, type parameters could be explicitly
marked as inputs, and otherwise assumed to be outputs. That would be
unfortunate, since associated types would *also* be outputs -- leaving the
language with two ways of specifying output types for traits.

But the second reason is for the library stabilization process:

* Since most existing uses of trait type parameters are intended as outputs,
  they should really be associated types instead. Making promises about these APIs
  as they currently stand risks locking the libraries into a design that will seem
  obsolete as soon as associated items are added. Again, this risk could probably
  be mitigated with a different, backwards-compatible associated item design, but
  at the cost of cruft in the language itself.

* The binary operator traits (e.g. `Add`) should be multidispatch. It does not
  seem possible to stabilize them *now* in a way that will support moving to
  multidispatch later.

* There are some thorny problems in the current libraries, such as the `_equiv`
  methods accumulating in `HashMap`, that can be solved using associated
  items. (See "Defaults" below for more on this specific example.) Additional
  examples include traits for error propagation and for conversion (to be
  covered in future RFCs). Adding these traits would improve the quality and
  consistency of our 1.0 library APIs.

# Detailed design

## Trait headers

Trait headers are written according to the following grammar:

```
TRAIT_HEADER =
  'trait' IDENT [ '<' INPUT_PARAMS '>' ] [ ':' BOUNDS ] [ WHERE_CLAUSE ]

INPUT_PARAMS = INPUT_TY { ',' INPUT_TY }* [ ',' ]
INPUT_PARAM  = IDENT [ ':' BOUNDS ]

BOUNDS = BOUND { '+' BOUND }* [ '+' ]
BOUND  = IDENT [ '<' ARGS '>' ]

ARGS   = INPUT_ARGS
       | OUTPUT_CONSTRAINTS
       | INPUT_ARGS ',' OUTPUT_CONSTRAINTS

INPUT_ARGS = TYPE { ',' TYPE }*

OUTPUT_CONSTRAINTS = OUTPUT_CONSTRAINT { ',' OUTPUT_CONSTRAINT }*
OUTPUT_CONSTRAINT  = IDENT '=' TYPE
```

**NOTE**: The grammar for `WHERE_CLAUSE` and `BOUND` is explained in detail in
  the subsection "Constraining associated types" below.

All type parameters to a trait are considered inputs, and can be used to select
an `impl`; conceptually, each distinct instantiation of the types yields a
distinct trait. More details are given in the section "The input/output type
distinction" below.

## Trait bodies: defining associated items

Trait bodies are expanded to include three new kinds of items: consts, types,
and lifetimes:

```
TRAIT = TRAIT_HEADER '{' TRAIT_ITEM* '}'
TRAIT_ITEM =
  ... <existing productions>
  | 'const' IDENT ':' TYPE [ '=' CONST_EXP ] ';'
  | 'type' IDENT [ ':' BOUNDS ] [ WHERE_CLAUSE ] [ '=' TYPE ] ';'
  | 'lifetime' LIFETIME_IDENT ';'
```

Traits already support associated functions, which had previously been called
"static" functions.

The `BOUNDS` and `WHERE_CLAUSE` on associated types are *obligations* for the
implementor of the trait, and *assumptions* for users of the trait:

```rust
trait Graph {
    type N: Show + Hash;
    type E: Show + Hash;
    ...
}

impl Graph for MyGraph {
    // Both MyNode and MyEdge must implement Show and Hash
    type N = MyNode;
    type E = MyEdge;
    ...
}

fn print_nodes<G: Graph>(g: &G) {
    // here, can assume G::N implements Show
    ...
}
```

### Namespacing/shadowing for associated types

Associated types may have the same name as existing types in scope, *except* for
type parameters to the trait:

```rust
struct Foo { ... }

trait Bar<Input> {
    type Foo; // this is allowed
    fn into_foo(self) -> Foo; // this refers to the trait's Foo

    type Input; // this is NOT allowed
}
```

By not allowing name clashes between input and output types,
keep open the possibility of later allowing syntax like:

```rust
Bar<Input=u8, Foo=uint>
```

where both input and output parameters are constrained by name. And anyway,
there is no use for clashing input/output names.

In the case of a name clash like `Foo` above, if the trait needs to refer to the
outer `Foo` for some reason, it can always do so by using a `type` alias
external to the trait.

### Defaults

Notice that associated consts and types both permit defaults, just as trait
methods and functions can provide defaults.

Defaults are useful both as a code reuse mechanism, and as a way to expand the
items included in a trait without breaking all existing implementors of the
trait.

Defaults for associated types, however, present an interesting question: can
default methods assume the default type? In other words, is the following
allowed?

```rust
trait ContainerKey : Clone + Hash + Eq {
    type Query: Hash = Self;
    fn compare(&self, other: &Query) -> bool { self == other }
    fn query_to_key(q: &Query) -> Self { q.clone() };
}

impl ContainerKey for String {
    type Query = str;
    fn compare(&self, other: &str) -> bool {
        self.as_slice() == other
    }
    fn query_to_key(q: &str) -> String {
        q.into_string()
    }
}

impl<K,V> HashMap<K,V> where K: ContainerKey {
    fn find(&self, q: &K::Query) -> &V { ... }
}
```

In this example, the `ContainerKey` trait is used to associate a "`Query`" type
(for lookups) with an owned key type. This resolves the thorny "equiv" problem
in `HashMap`, where the hash map keys are `String`s but you want to index the
hash map with `&str` values rather than `&String` values, i.e. you want the
following to work:

```rust
// H: HashMap<String, SomeType>
H.find("some literal")
```

rather than having to write

```rust
H.find(&"some literal".to_string())`
```

The current solution involves duplicating the API surface with `_equiv` methods
that use the somewhat subtle `Equiv` trait, but the associated type approach
makes it easy to provide a simple, single API that covers the same use cases.

The defaults for `ContainerKey` just assume that the owned key and lookup key
types are the same, but the default methods have to assume the default
associated types in order to work.

For this to work, it must *not* be possible for an implementor of `ContainerKey`
to override the default `Query` type while leaving the default methods in place,
since those methods may no longer typecheck.

We deal with this in a very simple way:

* If a trait implementor overrides any default associated types, they must also
  override *all* default functions and methods.

* Otherwise, a trait implementor can selectively override individual default
  methods/functions, as they can today.

## Trait implementations

Trait `impl` syntax is much the same as before, except that const, type, and
lifetime items are allowed:

```
IMPL_ITEM =
  ... <existing productions>
  | 'const' IDENT ':' TYPE '=' CONST_EXP ';'
  | 'type' IDENT' '=' 'TYPE' ';'
  | 'lifetime' LIFETIME_IDENT '=' LIFETIME_REFERENCE ';'
```

Any `type` implementation must satisfy all bounds and where clauses in the
corresponding trait item.

## Referencing associated items

Associated items are referenced through paths. The expression path grammar was
updated as part of [UFCS](https://github.com/rust-lang/rfcs/pull/132), but to
accommodate associated types and lifetimes we need to update the type path
grammar as well.

The full grammar is as follows:

```
EXP_PATH
  = EXP_ID_SEGMENT { '::' EXP_ID_SEGMENT }*
  | TYPE_SEGMENT { '::' EXP_ID_SEGMENT }+
  | IMPL_SEGMENT { '::' EXP_ID_SEGMENT }+
EXP_ID_SEGMENT   = ID [ '::' '<' TYPE { ',' TYPE }* '>' ]

TY_PATH
  = TY_ID_SEGMENT { '::' TY_ID_SEGMENT }*
  | TYPE_SEGMENT { '::' TY_ID_SEGMENT }*
  | IMPL_SEGMENT { '::' TY_ID_SEGMENT }+

TYPE_SEGMENT = '<' TYPE '>'
IMPL_SEGMENT = '<' TYPE 'as' TRAIT_REFERENCE '>'
TRAIT_REFERENCE = ID [ '<' TYPE { ',' TYPE * '>' ]
```

Here are some example paths, along with what they might be referencing

```rust
// Expression paths ///////////////////////////////////////////////////////////////

a::b::c         // reference to a function `c` in module `a::b`
a::<T1, T2>     // the function `a` instantiated with type arguments `T1`, `T2`
Vec::<T>::new   // reference to the function `new` associated with `Vec<T>`
<Vec<T> as SomeTrait>::some_fn
                // reference to the function `some_fn` associated with `SomeTrait`,
                //   as implemented by `Vec<T>`
T::size_of      // the function `size_of` associated with the type or trait `T`
<T>::size_of    // the function `size_of` associated with `T` _viewed as a type_
<T as SizeOf>::size_of
                // the function `size_of` associated with `T`'s impl of `SizeOf`

// Type paths /////////////////////////////////////////////////////////////////////

a::b::C         // reference to a type `C` in module `a::b`
A<T1, T2>       // type A instantiated with type arguments `T1`, `T2`
Vec<T>::Iter    // reference to the type `Iter` associated with `Vec<T>
<Vec<T> as SomeTrait>::SomeType
                // reference to the type `SomeType` associated with `SomeTrait`,
                //   as implemented by `Vec<T>`
```

### Ways to reference items

Next, we'll go into more detail on the meaning of each kind of path.

For the sake of discussion, we'll suppose we've defined a trait like the
following:

```rust
trait Container {
    type E;
    fn empty() -> Self;
    fn insert(&mut self, E);
    fn contains(&self, &E) -> bool where E: PartialEq;
    ...
}

impl<T> Container for Vec<T> {
    type E = T;
    fn empty() -> Vec<T> { Vec::new() }
    ...
}
```

#### Via an `ID_SEGMENT` prefix

##### When the prefix resolves to a type

The most common way to get at an associated item is through a type parameter
with a trait bound:

```rust
fn pick<C: Container>(c: &C) -> Option<&C::E> { ... }

fn mk_with_two<C>() -> C where C: Container, C::E = uint {
    let mut cont = C::empty();  // reference to associated function
    cont.insert(0);
    cont.insert(1);
    cont
}
```

For these references to be valid, the type parameter must be known to implement
the relevant trait:

```rust
// Knowledge via bounds
fn pick<C: Container>(c: &C) -> Option<&C::E> { ... }

// ... or equivalently,  where clause
fn pick<C>(c: &C) -> Option<&C::E> where C: Container { ... }

// Knowledge via ambient constraints
struct TwoContainers<C1: Container, C2: Container>(C1, C2);
impl<C1: Container, C2: Container> TwoContainers<C1, C2> {
    fn pick_one(&self) -> Option<&C1::E> { ... }
    fn pick_other(&self) -> Option<&C2::E> { ... }
}
```

Note that `Vec<T>::E` and `Vec::<T>::empty` are also valid type and function
references, respectively.

For cases like `C::E` or `Vec<T>::E`, the path begins with an `ID_SEGMENT`
prefix that itself resolves to a _type_: both `C` and `Vec<T>` are types.  In
general, a path `PREFIX::REST_OF_PATH` where `PREFIX` resolves to a type is
equivalent to using a `TYPE_SEGMENT` prefix `<PREFIX>::REST_OF_PATH`. So, for
example, following are all equivalent:

```rust
fn pick<C: Container>(c: &C) -> Option<&C::E> { ... }
fn pick<C: Container>(c: &C) -> Option<&<C>::E> { ... }
fn pick<C: Container>(c: &C) -> Option<&<<C>::E>> { ... }
```

The behavior of `TYPE_SEGMENT` prefixes is described in the next subsection.

##### When the prefix resolves to a trait

However, it is possible for an `ID_SEGMENT` prefix to resolve to a *trait*,
rather than a type. In this case, the behavior of an `ID_SEGMENT` varies from
that of a `TYPE_SEGMENT` in the following way:

```rust
// a reference Container::insert is roughly equivalent to:
fn trait_insert<C: Container>(c: &C, e: C::E);

// a reference <Container>::insert is roughly equivalent to:
fn object_insert<E>(c: &Container<E=E>, e: E);
```

That is, if `PREFIX` is an `ID_SEGMENT` that
resolves to a trait `Trait`:

* A path `PREFIX::REST` resolves to the item/path `REST` defined within
  `Trait`, while treating the type implementing the trait as a type parameter.

* A path `<PREFIX>::REST` treats `PREFIX` as a (DST-style) *type*, and is
  hence usable only with trait objects. See the
  [UFCS RFC](https://github.com/rust-lang/rfcs/pull/132) for more detail.

Note that a path like `Container::E`, while grammatically valid, will fail to
resolve since there is no way to tell which `impl` to use. A path like
`Container::empty`, however, resolves to a function roughly equivalent to:

```rust
fn trait_empty<C: Container>() -> C;
```

#### Via a `TYPE_SEGMENT` prefix

> The following text is *slightly changed* from the
> [UFCS RFC](https://github.com/rust-lang/rfcs/pull/132).

When a path begins with a `TYPE_SEGMENT`, it is a type-relative path. If this is
the complete path (e.g., `<int>`), then the path resolves to the specified
type. If the path continues (e.g., `<int>::size_of`) then the next segment is
resolved using the following procedure.  The procedure is intended to mimic
method lookup, and hence any changes to method lookup may also change the
details of this lookup.

Given a path `<T>::m::...`:

1. Search for members of inherent impls defined on `T` (if any) with
   the name `m`. If any are found, the path resolves to that item.

2. Otherwise, let `IN_SCOPE_TRAITS` be the set of traits that are in
   scope and which contain a member named `m`:
   - Let `IMPLEMENTED_TRAITS` be those traits from `IN_SCOPE_TRAITS`
     for which an implementation exists that (may) apply to `T`.
     - There can be ambiguity in the case that `T` contains type inference
       variables.
   - If `IMPLEMENTED_TRAITS` is not a singleton set, report an ambiguity
     error. Otherwise, let `TRAIT` be the member of `IMPLEMENTED_TRAITS`.
   - If `TRAIT` is ambiguously implemented for `T`, report an
     ambiguity error and request further type information.
   - Otherwise, rewrite the path to `<T as Trait>::m::...` and
     continue.

#### Via a `IMPL_SEGMENT` prefix

> The following text is *somewhat different* from the
> [UFCS RFC](https://github.com/rust-lang/rfcs/pull/132).

When a path begins with an `IMPL_SEGMENT`, it is a reference to an item defined
from a trait. Note that such paths must always have a follow-on member `m` (that
is, `<T as Trait>` is not a complete path, but `<T as Trait>::m` is).

To resolve the path, first search for an applicable implementation of `Trait`
for `T`. If no implementation can be found -- or the result is ambiguous -- then
report an error.  Note that when `T` is a type parameter, a bound `T: Trait`
guarantees that there is such an implementation, but does not count for
ambiguity purposes.

Otherwise, resolve the path to the member of the trait with the substitution
`Self => T` and continue.

This apparently straightforward algorithm has some subtle consequences, as
illustrated by the following example:

```rust
trait Foo {
    type T;
    fn as_T(&self) -> &T;
}

// A blanket impl for any Show type T
impl<T: Show> Foo for T {
    type T = T;
    fn as_T(&self) -> &T { self }
}

fn bounded<U: Foo>(u: U) where U::T: Show {
    // Here, we just constrain the associated type directly
    println!("{}", u.as_T())
}

fn blanket<U: Show>(u: U) {
    // the blanket impl applies to U, so we know that `U: Foo` and
    // <U as Foo>::T = U (and, of course, U: Show)
    println!("{}", u.as_T())
}

fn not_allowed<U: Foo>(u: U) {
    // this will not compile, since <U as Trait>::T is not known to
    // implement Show
    println!("{}", u.as_T())
}
```

This example includes three generic functions that make use of an associated
type; the first two will typecheck, while the third will not.

* The first case, `bounded`, places a `Show` constraint directly on the
  otherwise-abstract associated type `U::T`. Hence, it is allowed to assume that
  `U::T: Show`, even though it does not know the concrete implementation of
  `Foo` for `U`.

* The second case, `blanket`, places a `Show` constraint on the type `U`, which
  means that the blanket `impl` of `Foo` applies even though we do not know the
  *concrete* type that `U` will be. That fact means, moreover, that we can
  compute exactly what the associated type `U::T` will be, and know that it will
  satisfy `Show. Coherence guarantees that that the blanket `impl` is the only
  one that could apply to `U`. (See the section "Impl specialization" under
  "Unresolved questions" for a deeper discussion of this point.)

* The third case assumes only that `U: Foo`, and therefore nothing is known
  about the associated type `U::T`. In particular, the function cannot assume
  that `U::T: Show`.

The resolution rules also interact with instantiation of type parameters in an
intuitive way. For example:

```rust
trait Graph {
    type N;
    type E;
    ...
}

impl Graph for MyGraph {
    type N = MyNode;
    type E = MyEdge;
    ...
}

fn pick_node<G: Graph>(t: &G) -> &G::N {
    // the type G::N is abstract here
    ...
}

let G = MyGraph::new();
...
pick_node(G) // has type: <MyGraph as Graph>::N = MyNode
```

Assuming there are no blanket implementations of `Graph`, the `pick_node`
function knows nothing about the associated type `G::N`. However, a *client* of
`pick_node` that instantiates it with a particular concrete graph type will also
know the concrete type of the value returned from the function -- here, `MyNode`.

## Scoping of `trait` and `impl` items

Associated types are frequently referred to in the signatures of a trait's
methods and associated functions, and it is natural and convneient to refer to
them directly.

In other words, writing this:

```rust
trait Graph {
    type N;
    type E;
    fn has_edge(&self, &N, &N) -> bool;
    ...
}
```

is more appealing than writing this:

```rust
trait Graph {
    type N;
    type E;
    fn has_edge(&self, &Self::N, &Self::N) -> bool;
    ...
}
```

This RFC proposes to treat both `trait` and `impl` bodies (both
inherent and for traits) the same way we treat `mod` bodies: *all*
items being defined are in scope. In particular, methods are in scope
as UFCS-style functions:

```rust
trait Foo {
    type AssocType;
    lifetime 'assoc_lifetime;
    const ASSOC_CONST: uint;
    fn assoc_fn() -> Self;

    // Note: 'assoc_lifetime and AssocType in scope:
    fn method(&self, Self) -> &'assoc_lifetime AssocType;

    fn default_method(&self) -> uint {
        // method in scope UFCS-style, assoc_fn in scope
        let _ = method(self, assoc_fn());
        ASSOC_CONST // in scope
    }
}

// Same scoping rules for impls, including inherent impls:
struct Bar;
impl Bar {
    fn foo(&self) { ... }
    fn bar(&self) {
        foo(self); // foo in scope UFCS-style
        ...
    }
}
```

Items from super traits are *not* in scope, however. See
[the discussion on super traits below](#super-traits) for more detail.

These scope rules provide good ergonomics for associated types in
particular, and a consistent scope model for language constructs that
can contain items (like traits, impls, and modules). In the long run,
we should also explore imports for trait items, i.e. `use
Trait::some_method`, but that is out of scope for this RFC.

Note that, according to this proposal, associated types/lifetimes are *not* in
scope for the optional `where` clause on the trait header. For example:

```rust
trait Foo<Input>
    // type parameters in scope, but associated types are not:
    where Bar<Input, Self::Output>: Encodable {

    type Output;
    ...
}
```

This setup seems more intuitive than allowing the trait header to refer directly
to items defined within the trait body.

It's also worth noting that *trait-level* `where` clauses are never needed for
constraining associated types anyway, because associated types also have `where`
clauses. Thus, the above example could (and should) instead be written as
follows:

```rust
trait Foo<Input> {
    type Output where Bar<Input, Output>: Encodable;
    ...
}
```

## Constraining associated types

Associated types are not treated as parameters to a trait, but in some cases a
function will want to constrain associated types in some way. For example, as
explained in the Motivation section, the `Iterator` trait should treat the
element type as an output:

```rust
trait Iterator {
    type A;
    fn next(&mut self) -> Option<A>;
    ...
}
```

For code that works with iterators generically, there is no need to constrain
this type:

```rust
fn collect_into_vec<I: Iterator>(iter: I) -> Vec<I::A> { ... }
```

But other code may have requirements for the element type:

* That it implements some traits (bounds).
* That it unifies with a particular type.

These requirements can be imposed via `where` clauses:

```rust
fn print_iter<I>(iter: I) where I: Iterator, I::A: Show { ... }
fn sum_uints<I>(iter: I) where I: Iterator, I::A = uint { ... }
```

In addition, there is a shorthand for equality constraints:

```rust
fn sum_uints<I: Iterator<A = uint>>(iter: I) { ... }
```

In general, a trait like:

```rust
trait Foo<Input1, Input2> {
    type Output1;
    type Output2;
    lifetime 'a;
    ...
}
```

can be written in a bound like:

```
T: Foo<I1, I2>
T: Foo<I1, I2, Output1 = O1>
T: Foo<I1, I2, Output2 = O2>
T: Foo<I1, I2, Output1 = O1, Output2 = O2>
T: Foo<I1, I2, Output1 = O1, 'a = 'b, Output2 = O2>
```

The output constraints must come after all input arguments, but can appear in
any order.

Note that output constraints are allowed when referencing a trait in a *type* or
a *bound*, but not in an `IMPL_SEGMENT` path:

* As a type: `fn foo(obj: Box<Iterator<A = uint>>` is allowed.
* In a bound: `fn foo<I: Iterator<A = uint>>(iter: I)` is allowed.
* In an `IMPL_SEGMENT`: `<I as Iterator<A = uint>>::next` is *not* allowed.

The reason not to allow output constraints in `IMPL_SEGMENT` is that such paths
are references to a trait implementation that has already been determined -- it
does not make sense to apply additional constraints to the implementation when
referencing it.

Output constraints are a handy shorthand when using trait bounds, but they are a
*necessity* for trait objects, which we discuss next.

## Trait objects

When using trait objects, the `Self` type is "erased", so different types
implementing the trait can be used under the same trait object type:

```rust
impl Show for Foo { ... }
impl Show for Bar { ... }

fn make_vec() -> Vec<Box<Show>> {
    let f = Foo { ... };
    let b = Bar { ... };
    let mut v = Vec::new();
    v.push(box f as Box<Show>);
    v.push(box b as Box<Show>);
    v
}
```

One consequence of erasing `Self` is that methods using the `Self` type as
arguments or return values cannot be used on trait objects, since their types
would differ for different choices of `Self`.

In the model presented in this RFC, traits have additional input parameters
beyond `Self`, as well as associated types that may vary depending on all of the
input parameters. This raises the question: which of these types, if any, are
erased in trait objects?

The approach we take here is the simplest and most conservative: when using a
trait as a *type* (i.e., as a trait object), *all* input and output types must
be provided as part of the type. In other words, *only* the `Self` type is
erased, and all other types are specified statically in the trait object type.

Consider again the following example:

```rust
trait Foo<Input1, Input2> {
    type Output1;
    type Output2;
    lifetime 'a;
    ...
}
```

Unlike the case for static trait bounds, which do not have to specify any of the
associated types or lifetimes (but do have to specify the input types), trait
object types must specify all of the types:

```rust
fn consume_foo<T: Foo<I1, I2>>(t: T) // this is valid
fn consume_obj(t: Box<Foo<I1, I2>>)  // this is NOT valid

// but this IS valid:
fn consume_obj(t: Box<Foo<I1, I2, Output1 = O2, Output2 = O2, 'a = 'static>>)
```

With this design, it is clear that none of the non-`Self` types are erased as
part of trait objects. But it leaves wiggle room to relax this restriction
later on: trait object types that are not allowed under this design can be given
meaning in some later design.

## Inherent associated items

All associated items are also allowed in inherent `impl`s, so a definition like
the following is allowed:

```rust
struct MyGraph { ... }
struct MyNode { ... }
struct MyEdge { ... }

impl MyGraph {
    type N = MyNode;
    type E = MyEdge;

    // Note: associated types in scope, just as with trait bodies
    fn has_edge(&self, &N, &N) -> bool {
        ...
    }

    ...
}
```

Inherent associated items are referenced similarly to trait associated items:

```rust
fn distance(g: &MyGraph, from: &MyGraph::N, to: &MyGraph::N) -> uint { ... }
```

Note, however, that output constraints do not make sense for inherent outputs:

```rust
// This is *not* a legal type:
MyGraph<N = SomeNodeType>
```

## The input/output type distinction

When designing a trait that references some unknown type, you now have the
option of taking that type as an input parameter, or specifying it as an output
associated type. What are the ramifications of this decision?

### Coherence implications

Input types are used when determining which `impl` matches, even for the same
`Self` type:

```rust
trait Iterable1<A> {
    type I: Iterator<A>;
    fn iter(self) -> I;
}

// These impls have distinct input types, so are allowed
impl Iterable1<u8> for Foo { ... }
impl Iterable1<char> for Foo { ... }

trait Iterable2 {
    type A;
    type I: Iterator<A>;
    fn iter(self) -> I;
}

// These impls apply to a common input (Foo), so are NOT allowed
impl Iterable2 for Foo { ... }
impl Iterable2 for Foo { ... }
```

More formally, the *coherence* property is revised as follows:

- Given a trait and values for all its type parameters (inputs, including
  `Self`), there is at most one applicable `impl`.

In the [trait reform RFC](https://github.com/rust-lang/rfcs/pull/48), coherence
is guaranteed by maintaining two other key properties, which are revised as
follows:

*Orphan check*: Every implementation must meet one of
the following conditions:

1. The trait being implemented (if any) must be defined in the current crate.
2. At least one of the input type parameters (including but not
   necessarily `Self`) must meet the following grammar, where `C`
   is a struct or enum defined within the current crate:

       T = C
         | [T]
         | [T, ..n]
         | &T
         | &mut T
         | ~T
         | (..., T, ...)
         | X<..., T, ...> where X is not bivariant with respect to T

*Overlapping instances*: No two implementations can be instantiable
with the same set of types for the input type parameters.

See the [trait reform RFC](https://github.com/rust-lang/rfcs/pull/48) for more
discussion of these properties.

### Type inference implications

Finally, *output* type parameters can be inferred/resolved as soon as there is
a matching `impl` based on the input type parameters. Because of the
coherence property above, there can be at most one.

On the other hand, even if there is only one applicable `impl`, type inference
is *not* allowed to infer the input type parameters from it. This restriction
makes it possible to ensure *crate concatentation*: adding another crate may add
`impl`s for a given trait, and if type inference depended on the absence of such
`impl`s, importing a crate could break existing code.

In practice, these inference benefits can be quite valuable. For example, in the
`Add` trait given at the beginning of this RFC, the `Sum` output type is
immediately known once the input types are known, which can avoid the need for
type annotations.

## Limitations

The main limitation of associated items as presented here is about associated
*types* in particular. You might be tempted to write a trait like the following:

```rust
trait Iterable {
    type A;
    type I: Iterator<&'a A>; // what is the lifetime here?
    fn iter<'a>(&'a self) -> I;  // and how to connect it to self?
}
```

The problem is that, when implementing this trait, the return type `I` of `iter`
must generally depend on the *lifetime* of self. For example, the corresponding
method in `Vec` looks like the following:

```rust
impl<T> Vec<T> {
    fn iter(&'a self) -> Items<'a, T> { ... }
}
```

This means that, given a `Vec<T>`, there isn't a *single* type `Items<T>` for
iteration -- rather, there is a *family* of types, one for each input lifetime.
In other words, the associated type `I` in the `Iterable` needs to be
"higher-kinded": not just a single type, but rather a family:

```rust
trait Iterable {
    type A;
    type I<'a>: Iterator<&'a A>;
    fn iter<'a>(&self) -> I<'a>;
}
```

In this case, `I` is parameterized by a lifetime, but in other cases (like
`map`) an associated type needs to be parameterized by a type.

In general, such higher-kinded types (HKTs) are a much-requested feature for
Rust, and they would extend the reach of associated types. But the design and
implementation of higher-kinded types is, by itself, a significant investment.
The point of view of this RFC is that associated items bring the most important
changes needed to stabilize our existing traits (and add a few key others),
while HKTs will allow us to define important traits in the future but are not
necessary for 1.0.

### Encoding higher-kinded types

That said, it's worth pointing out that variants of higher-kinded types can be
encoded in the system being proposed here.

For example, the `Iterable` example above can be written in the following
somewhat contorted style:

```rust
trait IterableOwned {
    type A;
    type I: Iterator<A>;
    fn iter_owned(self) -> I;
}

trait Iterable {
    fn iter<'a>(&'a self) -> <&'a Self>::I where &'a Self: IterableOwned {
        IterableOwned::iter_owned(self)
    }
}
```

The idea here is to define a trait that takes, as input type/lifetimes
parameters, the parameters to any HKTs. In this case, the trait is implemented
on the type `&'a Self`, which includes the lifetime parameter.

We can in fact generalize this technique to encode arbitrary HKTs:

```rust
// The kind * -> *
trait TypeToType<Input> {
    type Output;
}
type Apply<Name, Elt> where Name: TypeToType<Elt> = Name::Output;

struct Vec_;
struct DList_;

impl<T> TypeToType<T> for Vec_ {
    type Output = Vec<T>;
}

impl<T> TypeToType<T> for DList_ {
    type Output = DList<T>;
}

trait Mappable
{
    type E;
    type HKT where Apply<HKT, E> = Self;

    fn map<F>(self, f: E -> F) -> Apply<HKT, F>;
}
```

While the above demonstrates the versatility of associated types and `where`
clauses, it is probably too much of a hack to be viable for use in `libstd`.

### Associated consts in generic code

There are some restrictions on uses of associated consts in generic code. These
might be loosened or removed in the future (see the related sub-sections in
"Unresolved questions" below).

 1. Values of constant expressions in match patterns cannot depend on a type
    parameter (by extension, neither can the types of such expressions). This
    restriction is necessary for exhaustiveness and reachability to be checked
    in generic code.

    Note that the dependence of a value on a type parameter may be indirect:

    ```rust
    enum MyEnum {
        Var1,
        Var2,
    }
    trait HasVar {
        const VAR: MyEnum;
    }
    fn do_something<T: HasVar>(x: MyEnum) {
        const y: MyEnum = <T>::VAR;
        // The following is forbidden because the value `y` depends on `T`.
        match x {
            y => { /* ... */ }
            _ => { /* ... */ }
        }
        // However, this is OK because the guard is not a part of the pattern.
        match x {
            z if z == y => { /* ... */ }
            _ => { /* ... */ }
        }
    }
    ```

 2. Array sizes that depend on type parameters cannot be compared for equality
    by type-checking, with one exception: if the expression for an array size
    comprises only a single reference to a constant item (or associated item),
    it will be considered equal to any other array size that refers to the same
    item, even if that item itself depends on the type parameters.

    For clarification, here are some examples. Assume that `T` is a type
    parameter in the outer scope, and that it is known to have an associated
    const `<T>::N` of type `usize`.

    ```rust
    // This is OK (but there are limitations to how x can be used).
    let x: [u8; <T>::N] = [0u8; <T>::N];
    // Equivalent to the above.
    let x = [0u8; <T>::N];
    // Neither of the following are allowed (type checking shouldn't have to
    // know anything about arithmetic).
    let x: [u8; 2 * <T>::N] = [0u8; <T>::N + <T>::N];
    let x: [u8; <T>::N + 1] = [0u8; 1 + <T>::N];
    // Still not allowed.
    let x: [u8; <T>::N + 1] = [0u8; <T>::N + 1];
    // Workaround for the expression above.
    const N_PLUS_1: usize = <T>::N + 1;
    let x: [u8; N_PLUS_1] = [0u8; N_PLUS_1];
    // Neither of the following are allowed.
    const ALIAS_N_PLUS_1: usize = N_PLUS_1;
    let x: [u8; N_PLUS_1] = [0u8; ALIAS_N_PLUS_1];
    const ALIAS_N: usize = <T>::N;
    let x: [u8; <T>::N] = [0u8; ALIAS_N];
    ```

# Staging

Associated lifetimes are probably not necessary for the 1.0 timeframe. While we
currently have a few traits that are parameterized by lifetimes, most of these
can go away once DST lands.

On the other hand, associated lifetimes are probably trivial to implement once
associated types have been implemented.

# Other interactions

## Interaction with implied bounds

As part of the
[implied bounds](http://smallcultfollowing.com/babysteps/blog/2014/07/06/implied-bounds/)
idea, it may be desirable for this:

```rust
fn pick_node<G>(g: &G) -> &<G as Graph>::N
```

to be sugar for this:

```rust
fn pick_node<G: Graph>(g: &G) -> &<G as Graph>::N
```

But this feature can easily be added later, as part of a general implied bounds RFC.

## Future-proofing: specialization of `impl`s

In the future, we may wish to relax the "overlapping instances" rule so that one
can provide "blanket" trait implementations and then "specialize" them for
particular types. For example:

```rust
trait Sliceable {
    type Slice;
    // note: not using &self here to avoid need for HKT
    fn as_slice(self) -> Slice;
}

impl<'a, T> Sliceable for &'a T {
    type Slice = &'a T;
    fn as_slice(self) -> &'a T { self }
}

impl<'a, T> Sliceable for &'a Vec<T> {
    type Slice = &'a [T];
    fn as_slice(self) -> &'a [T] { self.as_slice() }
}
```

But then there's a difficult question:

```
fn dice<A>(a: &A) -> &A::Slice where &A: Slicable {
    a // is this allowed?
}
```

Here, the blanket and specialized implementations provide incompatible
associated types. When working with the trait generically, what can we assume
about the associated type? If we assume it is the blanket one, the type may
change during monomorphization (when specialization takes effect)!

The RFC *does* allow generic code to "see" associated types provided by blanket
implementations, so this is a potential problem.

Our suggested strategy is the following. If at some later point we wish to add
specialization, traits would have to *opt in* explicitly. For such traits, we
would *not* allow generic code to "see" associated types for blanket
implementations; instead, output types would only be visible when all input
types were concretely known. This approach is backwards-compatible with the RFC,
and is probably a good idea in any case.

# Alternatives

## Multidispatch through tuple types

This RFC clarifies trait matching by making trait type parameters inputs to
matching, and associated types outputs.

A more radical alternative would be to *remove type parameters from traits*, and
instead support multiple input types through a separate multidispatch mechanism.

In this design, the `Add` trait would be written and implemented as follows:

```rust
// Lhs and Rhs are *inputs*
trait Add for (Lhs, Rhs) {
    type Sum; // Sum is an *output*
    fn add(&Lhs, &Rhs) -> Sum;
}

impl Add for (int, int) {
    type Sum = int;
    fn add(left: &int, right: &int) -> int { ... }
}

impl Add for (int, Complex) {
    type Sum = Complex;
    fn add(left: &int, right: &Complex) -> Complex { ... }
}
```

The `for` syntax in the trait definition is used for multidispatch traits, here
saying that `impl`s must be for pairs of types which are bound to `Lhs` and
`Rhs` respectively. The `add` function can then be invoked in UFCS style by
writing

```rust
Add::add(some_int, some_complex)
```

*Advantages of the tuple approach*:

- It does not force a distinction between `Self` and other input types, which in
  some cases (including binary operators like `Add`) can be artificial.

- Makes it possible to specify input types without specifying the trait:
  `<(A, B)>::Sum` rather than `<A as Add<B>>::Sum`.

*Disadvantages of the tuple approach*:

- It's more painful when you *do* want a method rather than a function.

- Requires `where` clauses when used in bounds: `where (A, B): Trait` rather
  than `A: Trait<B>`.

- It gives two ways to write single dispatch: either without `for`, or using
  `for` with a single-element tuple.

- There's a somewhat jarring distinction between single/multiple dispatch
  traits, making the latter feel "bolted on".

- The tuple syntax is unusual in acting as a binder of its types, as opposed to
  the `Trait<A, B>` syntax.

- Relatedly, the generics syntax for traits is immediately understandable (a
  family of traits) based on other uses of generics in the language, while the
  tuple notation stands alone.

- Less clear story for trait objects (although the fact that `Self` is the only
  erased input type in this RFC may seem somewhat arbitrary).

On balance, the generics-based approach seems like a better fit for the language
design, especially in its interaction with methods and the object system.

## A backwards-compatible version

Yet another alternative would be to allow trait type parameters to be either
inputs or outputs, marking the inputs with a keyword `in`:

```rust
trait Add<in Rhs, Sum> {
    fn add(&Lhs, &Rhs) -> Sum;
}
```

This would provide a way of adding multidispatch now, and then adding associated
items later on without breakage. If, in addition, output types had to come after
all input types, it might even be possible to migrate output type parameters
like `Sum` above into associated types later.

This is perhaps a reasonable fallback, but it seems better to introduce a clean
design with both multidispatch and associated items together.

# Unresolved questions

## Super traits

This RFC largely ignores super traits.

Currently, the implementation of super traits treats them identically to a
`where` clause that bounds `Self`, and this RFC does not propose to change
that. However, a follow-up RFC should clarify that this is the intended
semantics for super traits.

Note that this treatment of super traits is, in particular, consistent with the
proposed scoping rules, which do not bring items from super traits into scope in
the body of a subtrait; they must be accessed via `Self::item_name`.

## Equality constraints in `where` clauses

This RFC allows equality constraints on types for associated types, but does not
propose a similar feature for `where` clauses. That will be the subject of a
follow-up RFC.

## Multiple trait object bounds for the same trait

The design here makes it possible to write bounds or trait objects that mention
the same trait, multiple times, with different inputs:

```rust
fn mulit_add<T: Add<int> + Add<Complex>>(t: T) -> T { ... }
fn mulit_add_obj(t: Box<Add<int> + Add<Complex>>) -> Box<Add<int> + Add<Complex>> { ... }
```

This seems like a potentially useful feature, and should be unproblematic for
bounds, but may have implications for vtables that make it problematic for trait
objects. Whether or not such trait combinations are allowed will likely depend
on implementation concerns, which are not yet clear.

## Generic associated consts in match patterns

It seems desirable to allow constants that depend on type parameters in match
patterns, but it's not clear how to do so.

Looking at the `HasVar` example above, one possibility would be to simply treat
the first, forbidden match expression as syntactic sugar for the second, allowed
match expression that uses a pattern guard. This is simple to implement because
one can simply ignore the constant when performing exhaustiveness and
reachability checks. Unfortunately, this approach blurs the difference between
match patterns (which provide strict checks) and pattern guards (which are just
useful syntactic sugar), and it does not increase the expressiveness of the
language.

An alternative would be to allow `where` clauses to place constraints on
associated consts. If an associated const is known to be equal/unequal to some
other value (or in the case of integers, inside/outside a given range), this can
inform exhaustiveness and reachability checks. But this requires more design and
implementation work, and more syntax.

For now, we simply defer the question.

## Generic associated consts in array sizes

The above solution for type-checking array sizes is somewhat unsatisfactory. In
particular, it is counter-intuitive that neither of the following will type
check:

```rust
// Shouldn't this be OK?
const ALIAS_N: usize = <T>::N;
let x: [u8; <T>::N] = [0u8; ALIAS_N];
// This is likely to yield an embarrassing error message such as:
// "couldn't prove that `<T>::N + 1` is equal to `<T>::N + 1`"
let x: [u8; <T>::N + 1] = [0u8; <T>::N + 1];
```

A function like this is especially affected:

```rust
trait HasN {
    const N: usize;
}
fn foo<T: HasN>() -> [u8; <T>::N + 1] {
    // Can't be verified to be correct for the return type, and can't use the
    // intermediate const workaround due to scoping issues.
    [0u8; <T>::N + 1]
}
```

This can be worked around with type-level naturals that use associated consts to
produce array sizes, but this is syntactically a bit inelegant.

```rust
// Assume that `TypeAdd` and `One` are from a type-level naturals or similar
// library, and that `NAsTypeNatN` provides some way of translating the `N`
// on a `HasN` to a type compatible with that library.
trait HasN {
    const N: usize;
    type TypeNatN;
}
fn foo<T: HasN>() -> [u8; TypeAdd<<T>::TypeNatN, One>::AsUsize] {
    // Because the type `TypeAdd<<T>::TypeNatN, One>` can be verified to be
    // equal to itself in type checking, we know that the associated const
    // `AsUsize` below must be the same item as the `AsUsize` mentioned in the
    // return type above.
    [0u8; TypeAdd<<T>::NAsTypeNat, One>::AsUsize]
}
```

There are a variety of possible ways to address the above issues, including:

 - Implementing smarter handling of consts that are just aliases of other
   constant items.
 - Allowing `where` clauses to constrain some associated constants to be equal,
   to other expressions, and using this information in type checking.
 - Adding normalization with little or no awareness of arithmetic (e.g. allowing
   expressions that are exactly the same to be considered equal, or using only
   a very basic understanding of which operations are commutative and/or
   associative).
 - Adding new syntax and/or new capability to plugins to allow type-level
   naturals to be used with more ergonomic and clear syntax.
 - Implementing a dependent type system that provides built-in semantics for
   integer arithmetic at the type level, rather than implementing this in an
   external or standard library.
 - Using a full-fledged SMT solver.
 - Some other creative solutions not on this list.

While there are many ways to improve on the current design, and many of these
approaches are not mutually exclusive, much more work is needed to investigate
and implement a self-consistent, effective, and ideally intuitive set of
solutions.

Though admittedly not very satisfying at the moment, the current approach has
the advantage of being (arguably) a good minimalist design, allowing associated
consts to be used for array sizes in generic code now, but also allowing for any
of a number of improved systems to be implemented later.
