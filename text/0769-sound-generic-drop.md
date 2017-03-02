- Start Date: 2013-08-29
- RFC PR: [rust-lang/rfcs#769](https://github.com/rust-lang/rfcs/pull/769)
- Rust Issue: [rust-lang/rust#8861](https://github.com/rust-lang/rust/issues/8861)

# History

2015.09.18 -- This RFC was partially superceded by RFC 1238, which
removed the parametricity-based reasoning in favor of an attribute.

# Summary

Remove `#[unsafe_destructor]` from the Rust language.  Make it safe
for developers to implement `Drop` on type- and lifetime-parameterized
structs and enum (i.e. "Generic Drop") by imposing new rules on code
where such types occur, to ensure that the drop implementation cannot
possibly read or write data via a reference of type `&'a Data` where
`'a` could have possibly expired before the drop code runs.

Note: This RFC is describing a feature that has been long in the
making; in particular it was previously sketched in Rust [Issue #8861]
"New Destructor Semantics" (the source of the tongue-in-cheek "Start
Date" given above), and has a [prototype implementation] that is being
prepared to land.  The purpose of this RFC is two-fold:

 1. standalone documentation of the (admittedly conservative) rules
    imposed by the new destructor semantics, and

 2. elicit community feedback on the rules, both in the form they will
    take for 1.0 (which is relatively constrained) and the form they
    might take in the future (which allows for hypothetical language
    extensions).

[Issue #8861]: https://github.com/rust-lang/rust/issues/8861

[prototype implementation]: https://github.com/pnkfelix/rust/tree/77afdb70a1d4d5a20069f12412bfeda3ccd145bf

# Motivation

Part of Rust's design is rich use of Resource Acquisition Is
Initialization (RAII) patterns, which requires destructors: code
attached to certain types that runs only when a value of the type goes
out of scope or is otherwise deallocated. In Rust, the `Drop` trait is
used for this purpose.

Currently (as of Rust 1.0 alpha), a developer cannot implement `Drop`
on a type- or lifetime-parametric type (e.g. `struct Sneetch<'a>` or
`enum Zax<T>`) without attaching the `#[unsafe_destructor]` attribute
to it. The reason this attribute is required is that the current
implementation allows for such destructors to inject unsoundness
accidentally (e.g. reads from or writes to deallocated memory,
accessing data when its representation invariants are no longer
valid).

Furthermore, while some destructors can be implemented with no danger
of unsoundness, regardless of `T` (assuming that any `Drop`
implementation attached to `T` is itself sound), as soon as one wants
to interact with borrowed data within the `fn drop` code (e.g. access
a field `&'a StarOffMachine` from a value of type `Sneetch<'a>` ),
there is currently no way to enforce a rule that `'a` *strictly*
*outlive* the value itself. This is a huge gap in the language as it
stands: as soon as a developer attaches `#[unsafe_destructor]` to such
a type, it is imposing a subtle and *unchecked* restriction on clients
of that type that they will not ever allow the borrowed data to expire
first.

## Lifetime parameterization: the Sneetch example
[The Sneetch example]: #lifetime-parameterization-the-sneetch-example

If today Sylvester writes:

```rust
// opt-in to the unsoundness!
#![feature(unsafe_destructor)]

pub mod mcbean {
    use std::cell::Cell;

    pub struct StarOffMachine {
        usable: bool,
        dollars: Cell<u64>,
    }

    impl Drop for StarOffMachine {
        fn drop(&mut self) {
            let contents = self.dollars.get();
            println!("Dropping a machine; sending {} dollars to Sylvester.",
                     contents);
            self.dollars.set(0);
            self.usable = false;
        }
    }

    impl StarOffMachine {
        pub fn new() -> StarOffMachine {
            StarOffMachine { usable: true, dollars: Cell::new(0) }
        }
        pub fn remove_star(&self, s: &mut Sneetch) {
            assert!(self.usable,
                    "No different than a read of a dangling pointer.");
            self.dollars.set(self.dollars.get() + 10);
            s.has_star = false;
        }
    }

    pub struct Sneetch<'a> {
        name: &'static str,
        has_star: bool,
        machine: Cell<Option<&'a StarOffMachine>>,
    }

    impl<'a> Sneetch<'a> {
        pub fn new(name: &'static str) -> Sneetch<'a> {
            Sneetch {
                name: name,
                has_star: true,
                machine: Cell::new(None)
            }
        }

        pub fn find_machine(&self, m: &'a StarOffMachine) {
            self.machine.set(Some(m));
        }
    }

    #[unsafe_destructor]
    impl<'a> Drop for Sneetch<'a> {
        fn drop(&mut self) {
            if let Some(m) = self.machine.get() {
                println!("{} says ``before I die, I want to join my \
                          plain-bellied brethren.''", self.name);
                m.remove_star(self);
            }
        }
    }
}

fn unwary_client() {
    use mcbean::{Sneetch, StarOffMachine};
    let (s1, m, s2, s3); // (accommodate PR 21657)
    s1 = Sneetch::new("Sneetch One");
    m = StarOffMachine::new();
    s2 = Sneetch::new("Sneetch Two");
    s3 = Sneetch::new("Sneetch Zee");

    s1.find_machine(&m);
    s2.find_machine(&m);
    s3.find_machine(&m);
}

fn main() {
    unwary_client();
}
```

This compiles today; if you run it, it prints the following:

```
Sneetch Zee says ``before I die, I want to join my plain-bellied brethren.''
Sneetch Two says ``before I die, I want to join my plain-bellied brethren.''
Dropping a machine; sending 20 dollars to Sylvester.
Sneetch One says ``before I die, I want to join my plain-bellied brethren.''
thread '<main>' panicked at 'No different than a read of a dangling pointer.', <anon>:27
```

Explanation: In Sylvester's code, the `Drop` implementation for
`Sneetch` invokes a method on the borrowed reference in the field
`machine`. This implies there is an implicit restriction on an value
`s` of type `Sneetch<'a>`: the lifetime `'a` must *strictly outlive*
`s`.

(The example encodes this constraint in a dynamically-checked manner
via an explicit `usable` boolean flag that is only set to false in the
machine's own destructor; it is important to keep in mind that this is
just a method to illustrate the violation in a semi-reliable manner:
Using a machine after `usable` is set to false by its `fn drop` code
is analogous to dereferencing a `*mut T` that has been deallocated, or
similar soundness violations.)

Sylvester's API does not encode the constraint "`'a` must strictly
outlive the `Sneetch<'a>`" explicitly; Rust currently has no way of
expressing the constraint that one lifetime be strictly greater than
another lifetime or type (the form `'a:'b` only formally says that
`'a` must live *at least* as long as `'b`).

Thus, client code like that in `unwary_client` can inadvertantly set
up scenarios where Sylvester's code may break, and Sylvester might be
completely unaware of the vulnerability.

## Type parameterization: the problem of trait bounds
[The Zook example]: #type-parameterization-the-problem-of-trait-bounds

One might think that all instances of this problem can
be identified by the use of a lifetime-parametric `Drop` implementation,
such as `impl<'a> Drop for Sneetch<'a> { ..> }`

However, consider this trait and struct:

```rust
trait Button { fn push(&self); }
struct Zook<B: Button> { button: B, }
#[unsafe_destructor]
impl<B: Button> Drop for Zook<B> {
    fn drop(&mut self) { self.button.push(); }
}
```
In this case, it is not obvious that there is anything wrong here.

But if we continue the example:
```rust
struct Bomb { usable: bool }
impl Drop for Bomb { fn drop(&mut self) { self.usable = false; } }
impl Bomb { fn activate(&self) { assert!(self.usable) } }

enum B<'a> { HarmlessButton, BigRedButton(&'a Bomb) }
impl<'a> Button for B<'a> {
    fn push(&self) {
        if let B::BigRedButton(borrowed) = *self {
            borrowed.activate();
        }
    }
}

fn main() {
    let (mut zook, ticking);
    zook = Zook { button: B::HarmlessButton };
    ticking = Bomb { usable: true };
    zook.button = B::BigRedButton(&ticking);
}
```
Within the `zook` there is a hidden reference to borrowed data,
`ticking`, that is assigned the same lifetime as `zook` but that
will be dropped before `zook` is.

(These examples may seem contrived; see [Appendix A] for a far less
contrived example, that also illustrates how the use of borrowed data
can lie hidden behind type parameters.)

## The proposal

This RFC is proposes to fix this scenario, by having the compiler
ensure that types with destructors are only employed in contexts where
either any borrowed data with lifetime `'a` within the type either
strictly outlives the value of that type, or such borrowed data is
provably not accessible from any `Drop` implementation via a reference
of type `&'a`/`&'a mut`. This is the "Drop-Check" (aka `dropck`) rule.

# Detailed design

## The Drop-Check Rule
[The Drop-Check Rule]: #the-drop-check-rule

The Motivation section alluded to the compiler enforcing a new rule.
Here is a more formal statement of that rule:

Let `v` be some value (either temporary or named)
and `'a` be some lifetime (scope);
if the type of `v` owns data of type `D`, where
(1.) `D` has a lifetime- or type-parametric `Drop` implementation, and
(2.) the structure of `D` can reach a reference of type `&'a _`, and
(3.) either:

  * (A.) the `Drop impl` for `D` instantiates `D` at `'a`
         directly, i.e. `D<'a>`, or,

  * (B.) the `Drop impl` for `D` has some type parameter with a
         trait bound `T` where `T` is a trait that has at least
         one method,

then `'a` must strictly outlive the scope of `v`.

(Note: This rule is using two phrases that deserve further
elaboration and that are discussed further in sections that
follow: ["the type owns data of type `D`"][type-ownership]
and ["must strictly outlive"][strictly-outlives].)

(Note: When encountering a `D` of the form `Box<Trait+'b>`, we
conservatively assume that such a type has a `Drop` implementation
parametric in `'b`.)

This rule allows much sound existing code to compile without complaint
from `rustc`.  This is largely due to the fact that many `Drop`
implementations enjoy near-complete parametricity: They tend to not
impose any bounds at all on their type parameters, and thus the rule
does not apply to them.

At the same time, this rule catches the cases where a destructor could
possibly reference borrowed data via a reference of type `&'a _` or
`&'a mut_`. Here is why:

Condition (A.) ensures that a type like `Sneetch<'a>`
from [the Sneetch example] will only be
assigned to an expression `s` where `'a` strictly outlives `s`.

Condition (B.) catches cases like `Zook<B<'a>>` from
[the Zook example], where the destructor's interaction with borrowed
data is hidden behind a method call in the `fn drop`.

## Near-complete parametricity suffices

### Noncopy types

All non-`Copy` type parameters are (still) assumed to have a
destructor. Thus, one would be correct in noting that even a type
`T` with no bounds may still have one hidden method attached; namely,
its `Drop` implementation.

However, the drop implementation for `T` can only be called when
running the destructor for value `v` if either:

 1. the type of `v` owns data of type `T`, or

 2. the destructor of `v` constructs an instance of `T`.

In the first case, the Drop-Check rule ensures that `T` must satisfy
either Condition (A.) or (B.). In this second case, the freshly
constructed instance of `T` will only be able to access either
borrowed data from `v` itself (and thus such data will already have
lifetime that strictly outlives `v`) or data created during the
execution of the destructor.

### `Any` instances

All types implementing `Any` is forced to outlive `'static`. So one
should not be able to hide borrowed data behind the `Any` trait, and
therefore it is okay for the analysis to treat `Any` like a black box
whose destructor is safe to run (at least with respect to not
accessing borrowed data).

## Strictly outlives
[strictly-outlives]: #strictly-outlives

There is a notion of "strictly outlives" within the compiler
internals.  (This RFC is not adding such a notion to the language
itself; expressing "'a strictly outlives 'b" as an API constraint is
not a strict necessity at this time.)

The heart of the idea is this: we approximate the notion of "strictly
outlives" by the following rule: if a value `U` needs to strictly
outlive another value `V` with code extent `S`, we could just say that
`U` needs to live at least as long as the parent scope of `S`.

There are likely to be sound generalizations of the model given here
(and we will likely need to consider such to adopt future extensions
like Single-Entry-Multiple-Exit (SEME) regions, but that is out of
scope for this RFC).

In terms of its impact on the language, the main change has already
landed in the compiler; see [Rust PR 21657], which added
`CodeExtent::Remainder`, for more direct details on the implications
of that change written in a user-oriented fashion.

[Rust PR 21657]: https://github.com/rust-lang/rust/pull/21657

One important detail of the strictly-outlives relationship
that comes in part from [Rust PR 21657]:
All bindings introduced by a single `let` statement
are modeled as having the *same* lifetime.
In an example like
```rust
let a;
let b;
let (c, d);
...
```
`a` strictly outlives `b`, and `b` strictly outlives both `c` and `d`.
However, `c` and `d` are modeled as having the same lifetime; neither
one strictly outlives the other.
(Of course, during code execution, one of them will be dropped before
the other; the point is that when `rustc` builds its internal
model of the lifetimes of data, it approximates and assigns them
both the same lifetime.) This is an important detail,
because there are situations where one *must* assign the same
lifetime to two distinct bindings in order to allow them to
mutually refer to each other's data.

For more details on this "strictly outlives" model, see [Appendix B].

## When does one type own another
[type-ownership]: #when-does-one-type-own-another

The definition of the Drop-Check Rule used the phrase
"if the type owns data of type `D`".

This criteria is based on recursive descent of the
structure of an input type `E`.

 * If `E` itself has a Drop implementation that satisfies either
   condition (A.) or (B.) then add, for all relevant `'a`,
   the constraint that `'a` must outlive the scope of
   the value that caused the recursive descent.

 * Otherwise, if we have previously seen `E` during the descent
   then skip it (i.e. we assume a type has no destructor of interest
   until we see evidence saying otherwise).
   This check prevents infinite-looping when we
   encounter recursive references to a type, which can arise
   in e.g. `Option<Box<Type>>`.

 * Otherwise, if `E` is a struct (or tuple), for each of the struct's
   fields, recurse on the field's type (i.e., a struct owns its
   fields).

 * Otherwise, if `E` is an enum, for each of the enum's variants,
   and for each field of each variant, recurse on the field's type
   (i.e., an enum owns its fields).

 * Otherwise, if `E` is of the form `& T`, `&mut T`, `* T`, or `fn (T, ...) -> T`,
   then skip this `E`
   (i.e., references, native pointers, and bare functions do not own
   the types they refer to).

 * Otherwise, recurse on any immediate type substructure of `E`.
   (i.e., an instantiation of a polymorphic type `Poly<T_1, T_2>` is
   assumed to own `T_1` and `T_2`; note that structs and enums *do
   not* fall into this category, as they are handled up above; but
   this does cover cases like `Box<Trait<T_1, T_2>+'a>`).

### Phantom Data

The above definition for type-ownership is (believed to be) sound for
pure Rust programs that do not use `unsafe`, but it does not suffice
for several important types without some tweaks.

In particular, consider the implementation of `Vec<T>`:
as of "Rust 1.0 alpha":
```rust
pub struct Vec<T> {
    ptr: NonZero<*mut T>,
    len: uint,
    cap: uint,
}
```

According to the above definition, `Vec<T>` does not own `T`.
This is clearly wrong.

However, it generalizing the rule to say that `*mut T` owns `T` would
be too conservative, since there are cases where one wants to use
`*mut T` to model references to state that are not owned.

Therefore, we need some sort of marker, so that types like `Vec<T>`
can express that values of that type own instances of `T`.
The `PhantomData<T>` marker proposed by [RFC 738] ("Support variance
for type parameters") is a good match for this.
This RFC assumes that either [RFC 738] will be accepted,
or if necessary, this RFC will be amended so that it
itself adds the concept of `PhantomData<T>` to the language.
Therefore, as an additional special case to the criteria above
for when the type `E` owns data of type `D`, we include:

 * If `E` is `PhantomData<T>`, then recurse on `T`.

[RFC 738]: https://github.com/rust-lang/rfcs/pull/738

## Examples of changes imposed by the Drop-Check Rule

### Some cyclic structure is still allowed
[Cyclic structure still allowed]: #some-cyclic-structure-is-still-allowed

Earlier versions of the Drop-Check rule were quite conservative, to
the point where cyclic data would be disallowed in many contexts.
The Drop-Check rule presented in this RFC was crafted to try
to keep many existing useful patterns working.

In particular, cyclic structure is still allowed in many
contexts.  Here is one concrete example:

```rust
use std::cell::Cell;

#[derive(Show)]
struct C<'a> {
    v: Vec<Cell<Option<&'a C<'a>>>>,
}

impl<'a> C<'a> {
    fn new() -> C<'a> {
        C { v: Vec::new() }
    }
}

fn f() {
    let (mut c1, mut c2, mut c3);
    c1 = C::new();
    c2 = C::new();
    c3 = C::new();

    c1.v.push(Cell::new(None));
    c1.v.push(Cell::new(None));
    c2.v.push(Cell::new(None));
    c2.v.push(Cell::new(None));
    c3.v.push(Cell::new(None));
    c3.v.push(Cell::new(None));

    c1.v[0].set(Some(&c2));
    c1.v[1].set(Some(&c3));
    c2.v[0].set(Some(&c2));
    c2.v[1].set(Some(&c3));
    c3.v[0].set(Some(&c1));
    c3.v[1].set(Some(&c2));
}
```

In this code, each of the nodes { `c1`, `c2`, `c3` } contains a
reference to the two other nodes, and those references are stored in a
`Vec`.  Note that all of the bindings are introduced by a single
let-statement; this is to accommodate the region inference system
which wants to assign a single code extent to the `'a` lifetime, as
discussed in the [strictly-outlives] section.

Even though `Vec<T>` itself is defined as implementing `Drop`,
it puts no bounds on `T`, and therefore that `Drop` implementation is
ignored by the Drop-Check rule.

### Directly mixing cycles and `Drop` is rejected

[The Sneetch example] illustrates a scenario were borrowed data is
dropped while there is still an outstanding borrow that will be
accessed by a destructor.  In that particular example, one can easily
reorder the bindings to ensure that the `StarOffMachine` outlives all
of the sneetches.

But there are other examples that have no such resolution.  In
particular, graph-structured data where the destructor for each node
accesses the neighboring nodes in the graph; this simply cannot be
done soundly, because when there are cycles, there is no legal order in which to drop the nodes.

(At least, we cannot do it soundly without imperatively removing a
node from the graph as the node is dropped; but we are not going to
attempt to support verifying such an invariant as part of this RFC; to
my knowledge it is not likely to be feasible with type-checking based
static analyses).

In any case, we can easily show some code that will now start to be
rejected due to the Drop-Check rule: we take the same `C<'a>` example
of cyclic structure given above, but we now attach a `Drop`
implementation to `C<'a>`:

```rust
use std::cell::Cell;

#[derive(Show)]
struct C<'a> {
    v: Vec<Cell<Option<&'a C<'a>>>>,
}

impl<'a> C<'a> {
    fn new() -> C<'a> {
        C { v: Vec::new() }
    }
}

// (THIS IS NEW)
impl<'a> Drop for C<'a> {
    fn drop(&mut self) { }
}

fn f() {
    let (mut c1, mut c2, mut c3);
    c1 = C::new();
    c2 = C::new();
    c3 = C::new();

    c1.v.push(Cell::new(None));
    c1.v.push(Cell::new(None));
    c2.v.push(Cell::new(None));
    c2.v.push(Cell::new(None));
    c3.v.push(Cell::new(None));
    c3.v.push(Cell::new(None));

    c1.v[0].set(Some(&c2));
    c1.v[1].set(Some(&c3));
    c2.v[0].set(Some(&c2));
    c2.v[1].set(Some(&c3));
    c3.v[0].set(Some(&c1));
    c3.v[1].set(Some(&c2));
}
```

Now the addition of `impl<'a> Drop for C<'a>` changes
the results entirely;

The Drop-Check rule sees the newly added `impl<'a> Drop for C<'a>`,
which means that for every value of type `C<'a>`, `'a` must strictly
outlive the value. But in the binding
`let (mut c1, mut c2, mut c3)` , all three bindings are assigned
the same type `C<'scope_of_c1_c2_and_c3>`, where
`'scope_of_c1_c2_and_c3` does not strictly outlive any of the three.
Therefore this code will be rejected.

(Note: it is irrelevant that the `Drop` implementation is a no-op
above. The analysis does not care what the contents of that code are;
it solely cares about the public API presented by the type to its
clients.  After all, the `Drop` implementation for `C<'a>` could be
rewritten tomorrow to contain code that accesses the neighboring
nodes.

### Some temporaries need to be given names

Due to the way that `rustc` implements the [strictly-outlives]
relation in terms of code-extents, the analysis does not know in an
expression like `foo().bar().quux()` in what order the temporary
values `foo()` and `foo().bar()` will be dropped.

Therefore, the Drop-Check rule sometimes forces one to rewrite the
code so that it is apparent to the compiler that the value from
`foo()` will definitely outlive the value from `foo().bar()`.

Thus, on occasion one is forced to rewrite:
```rust
let q = foo().bar().quux();
...
```

as:
```rust
let foo = foo();
let q = foo.bar().quux()
...
```

or even sometimes as:
```rust
let foo = foo();
let bar = foo.bar();
let q = bar.quux();
...
```
depending on the types involved.

In practice, pnkfelix saw this arise most often
with code like this:

```rust
for line in old_io::stdin().lock().lines() {
    ...
}
```

Here, the result of `stdin()` is a `StdinReader`, which holds a
`RaceBox` in a `Mutex` behind an `Arc`.  The result of the `lock()`
method is a `StdinReaderGuard<'a>`, which owns a `MutexGuard<'a,
RaceBox>`.  The `MutexGuard` has a `Drop` implementation that is
parametric in `'a`; thus, the Drop-Check rule insists that the
lifetime assigned to `'a` strictly outlive the `MutexGuard`.

So, under this RFC, we rewrite the code like so:
```rust
let stdin = old_io::stdin();
for line in stdin.lock().lines() {
    ...
}
```

(pnkfelix acknowledges that this rewrite is unfortunate.  Potential
future work would be to further revise the code extent system so that
the compiler knows that the temporary from `stdin()` will outlive the
temporary from `stdin().lock()`.  However, such a change to the
code extents could have unexpected fallout, analogous to the
fallout that was associated with [Rust PR 21657].)

### Mixing acyclic structure and `Drop` is sometimes rejected

This is an example of sound code, accepted today, that is
unfortunately rejected by the Drop-Check rule (at least in pnkfelix's
prototype):

```rust
#![feature(unsafe_destructor)]

use std::cell::Cell;

#[derive(Show)]
struct C<'a> {
    f: Cell<Option<&'a C<'a>>>,
}

impl<'a> C<'a> {
    fn new() -> C<'a> {
        C { f: Cell::new(None), }
    }
}

// force dropck to care about C<'a>
#[unsafe_destructor]
impl<'a> Drop for C<'a> {
    fn drop(&mut self) { }
}

fn f() {
    let c2;
    let mut c1;

    c1 = C::new();
    c2 = C::new();

    c1.f.set(Some(&c2));
}

fn main() {
    f();
}
```

In principle this should work, since `c1` and `c2` are assigned to
distinct code extents, and `c1` will be dropped before `c2`.  However,
in the prototype, the region inference system is determining that the
lifetime `'a` in `&'a C<'a>` (from the `c1.f.set(Some(&c2));`
statement) needs to cover the whole block, rather than just the block
remainder extent that is actually covered by the `let c2;`.

(This may just be a bug somewhere in the prototype, but for the time
being pnkfelix is going to assume that it will be a bug that this RFC
is forced to live with indefinitely.)

## Unsound APIs need to be revised or removed entirely
[Unsound APIs]: #unsound-apis-that-need-to-be-revised-or-removed-entirely

While the Drop-Check rule is designed to ensure that safe Rust code is
sound in its use of destructors, it cannot assure us that unsafe code
is sound. It is the responsibility of the author of unsafe code to
ensure it does not perform unsound actions; thus, we need to audit our
own API's to ensure that the standard library is not providing
functionality that circumvents the Drop-Check rule.

The most obvious instance of this is the `arena` crate: in particular:
one can use an instance of `arena::Arena` to create cyclic graph
structure where each node's destructor accesses (via `&_` references)
its neighboring nodes.

Here is a version of our running `C<'a>` example
(where we now do something interesting the destructor for `C<'a>`)
that demonstrates the problem:

Example:
```rust
extern crate arena;

use std::cell::Cell;

#[derive(Show)]
struct C<'a> {
    name: &'static str,
    v: Vec<Cell<Option<&'a C<'a>>>>,
    usable: bool,
}

impl<'a> Drop for C<'a> {
    fn drop(&mut self) {
        println!("dropping {}", self.name);
        for neighbor in self.v.iter().map(|v|v.get()) {
            if let Some(neighbor) = neighbor {
                println!("  {} checking neighbor {}",
                         self.name, neighbor.name);
                assert!(neighbor.usable);
            }
        }
        println!("done dropping {}", self.name);
        self.usable = false;

    }
}

impl<'a> C<'a> {
    fn new(name: &'static str) -> C<'a> {
        C { name: name, v: Vec::new(), usable: true }
    }
}

fn f() {
    use arena::Arena;
    let arena = Arena::new();
    let (c1, c2, c3);

    c1 = arena.alloc(|| C::new("c1"));
    c2 = arena.alloc(|| C::new("c2"));
    c3 = arena.alloc(|| C::new("c3"));

    c1.v.push(Cell::new(None));
    c1.v.push(Cell::new(None));
    c2.v.push(Cell::new(None));
    c2.v.push(Cell::new(None));
    c3.v.push(Cell::new(None));
    c3.v.push(Cell::new(None));

    c1.v[0].set(Some(c2));
    c1.v[1].set(Some(c3));
    c2.v[0].set(Some(c2));
    c2.v[1].set(Some(c3));
    c3.v[0].set(Some(c1));
    c3.v[1].set(Some(c2));
}
```

Calling `f()` results in the following printout:
```
dropping c3
  c3 checking neighbor c1
  c3 checking neighbor c2
done dropping c3
dropping c1
  c1 checking neighbor c2
  c1 checking neighbor c3
thread '<main>' panicked at 'assertion failed: neighbor.usable', ../src/test/compile-fail/dropck_untyped_arena_cycle.rs:19
```

This is unsound. It should not be possible to express such a
scenario without using `unsafe` code.

This RFC suggests that we revise the `Arena` API by adding a phantom
lifetime parameter to its type, and bound the values the arena
allocates by that phantom lifetime, like so:
```rust
pub struct Arena<'longer_than_self> {
    _invariant: marker::InvariantLifetime<'longer_than_self>,
    ...
}

impl<'longer_than_self> Arena<'longer_than_self> {
    pub fn alloc<T:'longer_than_self, F>(&self, op: F) -> &mut T
        where F: FnOnce() -> T {
        ...
    }
}
```
Admittedly, this is a severe limitation, since it forces the data
allocated by the Arena to store only references to data that strictly
outlives the arena, regardless of whether the allocated data itself
even has a destructor. (I.e., `Arena` would become much weaker than
`TypedArena` when attempting to work with cyclic structures).
(pnkfelix knows of no way to fix this without adding further extensions
to the language, e.g. some way to express "this type's destructor accesses
none of its borrowed data", which is out of scope for this RFC.)

Alternatively, we could just deprecate the `Arena` API, (which is not
marked as stable anyway.

The example given here can be adapted to other kinds of backing
storage structures, in order to double-check whether the API is likely
to be sound or not.  For example, the `arena::TypedArena<T>` type
appears to be sound (as long as it carries `PhantomData<T>` just like
`Vec<T>` does). In particular, when one ports the above example to use
`TypedArena` instead of `Arena`, it is statically rejected by `rustc`.

## The final goal: remove #[unsafe_destructor]

Once all of the above pieces have landed, lifetime- and
type-parameterized `Drop` will be safe, and thus we will be able to
remove `#[unsafe_destructor]`!

# Drawbacks

* The Drop-Check rule is a little complex, and does disallow some
  sound code that would compile today.

* The change proposed in this RFC places restrictions on uses of types
  with attached destructors, but provides no way for a type `Foo<'a>` to
  state as part of its public interface that its drop implementation
  will not read from any borrowed data of lifetime `'a`. (Extending the
  language with such a feature is potential future work, but is out of
  scope for this RFC.)

* Some useful interfaces are going to be disallowed by this RFC.
  For example, the RFC recommends that the current `arena::Arena`
  be revised or simply deprecated, due to its unsoundness.
  (If desired, we could add an `UnsafeArena` that continues
  to support the current `Arena` API with the caveat that its users need to
  *manually* enforce the constraint that the destructors do not access
  data that has been already dropped. But again, that decision is out
  of scope for this RFC.)

# Alternatives

We considered simpler versions of [the Drop-Check rule]; in
particular, an earlier version of it simply said that if the type of
`v` owns any type `D` that implements `Drop`, then for any lifetime
`'a` that `D` refers to, `'a` must strictly outlive the scope of `v`,
because the destructor for `D` might hypothetically access borrowed
data of lifetime `'a`.

 * This rule is simpler in the sense that it more obviously sound.

 * But this rule disallowed far more code; e.g.  the [Cyclic structure
   still allowed] example was rejected under this more naive rule,
   because `C<'a>` owns D = `Vec<Cell<Option<&'a C<'a>>>>`, and this
   particular D refers to `'a`.

----

Sticking with the current `#[unsafe_destructor]` approach to lifetime-
and type-parametric types that implement `Drop` is not really tenable;
we need to do something (and we have been planning to do something
like this RFC for over a year).

# Unresolved questions

* Is the Drop-Check rule provably sound?  pnkfelix has based his
  argument on informal reasoning about parametricity, but it would be
  good to put forth a more formal argument.  (And in the meantime,
  pnkfelix invites the reader to try to find holes in the rule,
  preferably with concrete examples that can be fed into the
  prototype.)

* How much can covariance help with some of the lifetime issues?

  See in particular [Rust Issue 21198] "new scoping rules for safe
  dtors may benefit from variance on type params"

[Rust Issue 21198]: https://github.com/rust-lang/rust/issues/21198

  Before adding Condition (B.) to [the Drop-Check Rule], it seemed
  like enabling covariance in more standard library types was going to
  be very important for landing this work.  And even now, it is
  possible that covariance could still play an important role.
  But nonetheless, there are some API's whose current form is fundamentally
  incompatible with covariance; e.g. the current `TypedArena<T>` API
  is fundamentally invariant with respect to `T`.

# Appendices

## Appendix A: Why and when would Drop read from borrowed data
[Appendix A]: #appendix-a-why-and-when-would-drop-read-from-borrowed-data

Here is a story, about two developers, Julia and Kurt, and the code
they hacked on.

Julia inherited some code, and it is misbehaving.  It appears like
key/value entries that the code inserts into the standard library's
`HashMap` are not always retrievable from the map. Julia's current
hypothesis is that something is causing the keys' computed hash codes
to change dynamically, sometime after the entries have been inserted
into the map (but it is not obvious when or if this change occurs, nor
what its source might be). Julia thinks this hypothesis is plausible,
but does not want to audit all of the key variants for possible causes
of hash code corruption until after she has hard evidence confirming
the hypothesis.

Julia writes some code that walks a hash map's internals and checks
that all of the keys produce a hash code that is consistent with their
location in the map.  However, since it is not clear when the keys'
hash codes are changing, it is not clear where in the overall code
base she should add such checks. (The hash map is sufficiently large
that she cannot simply add calls to do this consistency check
everywhere.)

However, there is one spot in the control flow that is a clear
contender: if the check is run right before the hash map is dropped,
then that would surely be sometime after the hypothesized corruption
had occurred.  In other words, a destructor for the hash map seems
like a good place to start; Julia could make her own local copy of the
hash map library and add this check to a `impl<K,V,S> Drop for
HashMap<K,V,S> { ... }` implementation.

In this new destructor code, Julia needs to invoke the hash-code
method on `K`.  So she adds the bound `where K: Eq + Hash<H>` to her
`HashMap` and its `Drop` implementation, along with the corresponding
code to walk the table's entries and check that the hash codes for all
the keys matches their position in the table.

Using this, Julia manages confirms her hypothesis (yay).  And since it
was a reasonable amount of effort to do this experiment, she puts this
variation of `HashMap` up on `crates.io`, calling it the
`CheckedHashMap` type.

Sometime later, Kurt pulls a copy of `CheckHashMap` off of
`crates.io`, and he happens to write some code that looks like this:

```rust
fn main() {
    #[derive(PartialEq, Eq, Hash, Debug)]
    struct Key<'a> { name: &'a str }

    {
        let (key, mut map, name) : (Key, CheckedHashMap<&Key, String>, String);
        name = format!("k1");
        map = CheckedHashMap::new();
        key = Key { name: &*name };
        map.map.insert(&key, format!("Value for k1"));
    }
}
```

And, kaboom: when the map goes out of scope, the destructor for
`CheckedHashMap` attempts to compute a hashcode on a reference to
`key` that may not still be valid, and even if `key` is still valid,
it holds a reference to a slice of name that likewise may not still be
valid.

This illustrates a case where one might legitimately mix destructor
code with borrowed data.  (Is this example any less contrived than
[the Sneetch example]? That is in the eye of the beholder.)

## Appendix B: strictly-outlives details
[Appendix B]: #appendix-b-strictly-outlives-details

The rest of this section gets into some low-level details of parts of
how `rustc` is implemented, largely because the changes described here
do have an impact on what results the `rustc` region inference system
produces (or fails to produce). It serves mostly to explain (1.) why
[Rust PR 21657] was implemented, and (2.) why one may sometimes see
indecipherable region-inference errors.

### Review: Code Extents

(Nothing here is meant to be new; its just providing context for the
next subsection.)

Every Rust expression evaluates to a value `V` that is either placed
into some location with an associated lifetime such as `'l`, or `V` is
associated with a block of code that statically delimits the `V`'s
runtime extent (i.e. we know from the function's text where `V` will
be dropped). In the `rustc` source, the blocks of code are sometimes
called "scopes" and sometimes "code extents"; I will try to stick to
the latter term here, since the word "scope" is terribly overloaded.

Currently, the code extents in Rust are arranged into a tree hierarchy
structured similarly to the abstract syntax tree; for any given code
extent, the compiler can ask for its parent in this hierarchy.

Every Rust expression `E` has an associated "terminating extent"
somewhere in its chain of parent code extents; temporary values
created during the execution of `E` are stored at stack locations
managed by `E`'s terminating extent. When we hit the end of the
terminating extent, all such temporaries are dropped.

An example of a terminating extent: in a let-statement like:
```rust
let <pat> = <expr>;
```
the terminating extent of `<expr>` is the let-statement itself. So in
an example like:
```rust
let a1 = input.f().g();`
...
```
there is a temporary value returned from `input.f()`, and it will live
until the end of the let statement, but not into the subsequent code
represented by `...`.  (The value resulting from `input.f().g()`, on
the other hand, will be stored in `a1` and lives until the end of the
block enclosing the let statement.)

(It is not important to this RFC to know the full set of rules
dictating which parent expressions are deemed terminating extents; we
just will assume that these things do exist.)

For any given code extent `S`, the parent code extent `P` of `S`, if
it exists, potentially holds bits of code that will execute after `S`
is done.  Any cleanup code for any values assigned to `P` will only
run after we have finished with *all* code associated with `S`.

### A problem with 1.0 alpha code extents

So, with the above established, we have a hint at how to express that
a lifetime `'a` needs to strictly outlive a particular code extent `S`:
simply say that `'a` needs to live at least long as `P`.

However, this is a little too simplistic, at least for the Rust
compiler circa Rust 1.0 alpha. The main problem is that all the
bindings established by let statements in a block are assigned the
same code extent.

This, combined with our simplistic definition, yields real problems.
For example, in:

```rust
{
    use std::fmt;
    #[derive(Debug)] struct DropLoud<T:fmt::Debug>(&'static str, T);
    impl<T:fmt::Debug> Drop for DropLoud<T> {
        fn drop(&mut self) { println!("dropping {}:{:?}", self.0, self.1); }
    }

    let c1 = DropLoud("c1", 1);
    let c2 = DropLoud("c2", &c1);
}
```

In principle, the code above is legal: `c2` will be dropped before
`c1` is, and thus it is okay that `c2` holds a borrowed reference to
`c1` that will be read when `c2` is dropped (indirectly via the
`fmt::Debug` implementation.

However, with the structure of code extents as of Rust 1.0 alpha, `c1`
and `c2` are both given the same code extent: that of the block
itself.  Thus in that context, this definition of "strictly outlives"
indicates that `c1` does *not* strictly outlive `c2`, because `c1`
does not live at least as long as the parent of the block; it only
lives until the end of the block itself.

This illustrates why "All the bindings established by let statements
in a block are assigned the same code extent" is a problem

### Block Remainder Code Extents

The solution proposed here (motivated by experience with the
prototype) is to introduce finer-grained code extents.  This solution
is essentially [Rust PR 21657], which has already landed in `rustc`.
(That is in part why this is merely an appendix, rather than part of
the body of the RFC itself.)

The code extents remain in a tree-hierarchy, but there are now extra
entries in the tree, which provide the foundation for a more precise
"strictly outlives" relation.

We introduce a new code extent, called a "block remainder" extent, for
every let statement in a block, representing the suffix of the block
covered by the bindings in that let statement.

For example, given `{ let (a, b) = EXPR_1; let c = EXPR_2; ... }`,
which previously had a code extent structure like:
```
{ let (a, b) = EXPR_1; let c = EXPR_2; ... }
               +----+          +----+
  +------------------+ +-------------+
+------------------------------------------+
```
so the parent extent of each let statement was the whole block.

But under the new rules, there are two new block remainder extents
introduced, with this structure:

```
{  let (a, b) = EXPR_1;  let c = EXPR_2; ...  }
                +----+           +----+
   +------------------+  +-------------+
                        +-------------------+   <-- new: block remainder 2
  +------------------------------------------+  <-- new: block remainder 1
+---------------------------------------------+
```

The first let-statement introduces a block remainder extent that
covers the lifetime for `a` and `b`.  The second let-statement
introduces a block remainder extent that covers the lifetime for `c`.

Each let-statement continues to be the terminating extent for its
initializer expression.  But now, the parent of the extent of the
second let statement is a block remainder extent ("block remainder
2"), and, importantly, the parent of block remainder 2 is another
block remainder extent ("block remainder 1").  This way, we precisely
represent the lifetimes of the named values bound by each let
statement, and know that `a` and `b` both strictly outlive `c`
as well as the temporary values created during evaluation of
`EXPR_2`.
Likewise, `c` strictly outlives the bindings and temporaries created
in the `...` that follows it.

### Why stop at let-statements?

This RFC does *not* propose that we attempt to go further and track
the order of destruction of the values bound by a *single* let
statement.

Such an experiment could be made part of future work, but for now, we
just continue to assign `a` and `b` to the same scope; the compiler
does not attempt to reason about what order they will be dropped in,
and thus we cannot for example reference data borrowed from `a` in any
destructor code for `b`.

The main reason that we do not want to attempt to produce even finer
grain scopes, at least not right now, is that there are scenarios
where it is *important* to be able to assign the same region to two
distinct pieces of data; in particular, this often arises when one
wants to build cyclic structure, as discussed in
[Cyclic structure still allowed].
