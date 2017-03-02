- Feature Name: const_fn
- Start Date: 2015-02-25
- RFC PR: [rust-lang/rfcs#911](https://github.com/rust-lang/rfcs/pull/911)
- Rust Issue: [rust-lang/rust#24111](https://github.com/rust-lang/rust/issues/24111)

# Summary

Allow marking free functions and inherent methods as `const`, enabling them to be
called in constants contexts, with constant arguments.

# Motivation

As it is right now, `UnsafeCell` is a stabilization and safety hazard: the field
it is supposed to be wrapping is public. This is only done out of the necessity
to initialize static items containing atomics, mutexes, etc. - for example:

```rust
#[lang="unsafe_cell"]
struct UnsafeCell<T> { pub value: T }
struct AtomicUsize { v: UnsafeCell<usize> }
const ATOMIC_USIZE_INIT: AtomicUsize = AtomicUsize {
    v: UnsafeCell { value: 0 }
};
```

This approach is fragile and doesn't compose well - consider having to initialize
an `AtomicUsize` static with `usize::MAX` - you would need a `const` for each
possible value.

Also, types like `AtomicPtr<T>` or `Cell<T>` have no way *at all* to initialize
them in constant contexts, leading to overuse of `UnsafeCell` or `static mut`,
disregarding type safety and proper abstractions.

During implementation, the worst offender I've found was `std::thread_local`:
all the fields of `std::thread_local::imp::Key` are public, so they can be
filled in by a macro - and they're also marked "stable" (due to the lack of
stability hygiene in macros).

A pre-RFC for the removal of the dangerous (and oftenly misued) `static mut`
received positive feedback, but only under the condition that abstractions
could be created and used in `const` and `static` items.

Another concern is the ability to use certain intrinsics, like `size_of`, inside
constant expressions, including fixed-length array types. Unlike keyword-based
alternatives, `const fn` provides an extensible and composable building block
for such features.

The design should be as simple as it can be, while keeping enough functionality
to solve the issues mentioned above.

The intention of this RFC is to introduce a minimal change that
enables safe abstraction resembling the kind of code that one writes
outside of a constant. Compile-time pure constants (the existing
`const` items) with added parametrization over types and values
(arguments) should suffice.

This RFC explicitly does not introduce a general CTFE mechanism. In
particular, conditional branching and virtual dispatch are still not
supported in constant expressions, which imposes a severe limitation
on what one can express.

# Detailed design

Functions and inherent methods can be marked as `const`:
```rust
const fn foo(x: T, y: U) -> Foo {
    stmts;
    expr
}
impl Foo {
    const fn new(x: T) -> Foo {
        stmts;
        expr
    }

    const fn transform(self, y: U) -> Foo {
        stmts;
        expr
    }
}
```

Traits, trait implementations and their methods cannot be `const` - this
allows us to properly design a constness/CTFE system that interacts well
with traits - for more details, see *Alternatives*.

Only simple by-value bindings are allowed in arguments, e.g. `x: T`. While
by-ref bindings and destructuring can be supported, they're not necessary
and they would only complicate the implementation.

The body of the function is checked as if it were a block inside a `const`:
```rust
const FOO: Foo = {
    // Currently, only item "statements" are allowed here.
    stmts;
    // The function's arguments and constant expressions can be freely combined.
    expr
}
```

As the current `const` items are not formally specified (yet), there is a need
to expand on the rules for `const` values (pure compile-time constants), instead
of leaving them implicit:
* the set of currently implemented expressions is: primitive literals, ADTs
(tuples, arrays, structs, enum variants), unary/binary operations on primitives,
casts, field accesses/indexing, capture-less closures, references and blocks
(only item statements and a tail expression)
* no side-effects (assignments, non-`const` function calls, inline assembly)
* struct/enum values are not allowed if their type implements `Drop`, but
this is not transitive, allowing the (perfectly harmless) creation of, e.g.
`None::<Vec<T>>` (as an aside, this rule could be used to allow `[x; N]` even
for non-`Copy` types of `x`, but that is out of the scope of this RFC)
* references are trully immutable, no value with interior mutability can be placed
behind a reference, and mutable references can only be created from zero-sized
values (e.g. `&mut || {}`) - this allows a reference to be represented just by
its value, with no guarantees for the actual address in memory
* raw pointers can only be created from an integer, a reference or another raw
pointer, and cannot be dereferenced or cast back to an integer, which means any
constant raw pointer can be represented by either a constant integer or reference
* as a result of not having any side-effects, loops would only affect termination,
which has no practical value, thus remaining unimplemented
* although more useful than loops, conditional control flow (`if`/`else` and
`match`) also remains unimplemented and only `match` would pose a challenge
* immutable `let` bindings in blocks have the same status and implementation
difficulty as `if`/`else` and they both suffer from a lack of demand (blocks
were originally introduced to `const`/`static` for scoping items used only in
the initializer of a global).

For the purpose of rvalue promotion (to static memory), arguments are considered
potentially varying, because the function can still be called with non-constant
values at runtime.

`const` functions and methods can be called from any constant expression:
```rust
// Standalone example.
struct Point { x: i32, y: i32 }

impl Point {
    const fn new(x: i32, y: i32) -> Point {
        Point { x: x, y: y }
    }

    const fn add(self, other: Point) -> Point {
        Point::new(self.x + other.x, self.y + other.y)
    }
}

const ORIGIN: Point = Point::new(0, 0);

const fn sum_test(xs: [Point; 3]) -> Point {
    xs[0].add(xs[1]).add(xs[2])
}

const A: Point = Point::new(1, 0);
const B: Point = Point::new(0, 1);
const C: Point = A.add(B);
const D: Point = sum_test([A, B, C]);

// Assuming the Foo::new methods used here are const.
static FLAG: AtomicBool = AtomicBool::new(true);
static COUNTDOWN: AtomicUsize = AtomicUsize::new(10);
#[thread_local]
static TLS_COUNTER: Cell<u32> = Cell::new(1);
```

Type parameters and their bounds are not restricted, though trait methods cannot
be called, as they are never `const` in this design. Accessing trait methods can
still be useful - for example, they can be turned into function pointers:
```rust
const fn arithmetic_ops<T: Int>() -> [fn(T, T) -> T; 4] {
    [Add::add, Sub::sub, Mul::mul, Div::div]
}
```

`const` functions can also be unsafe, allowing construction of types that require
invariants to be maintained (e.g. `std::ptr::Unique` requires a non-null pointer)
```rust
struct OptionalInt(u32);
impl OptionalInt {
    /// Value must be non-zero
    const unsafe fn new(val: u32) -> OptionalInt {
        OptionalInt(val)
    }
}
```

# Drawbacks

* A design that is not conservative enough risks creating backwards compatibility
hazards that might only be uncovered when a more extensive CTFE proposal is made,
after 1.0.

# Alternatives

* While not an alternative, but rather a potential extension, I want to point
out there is only way I could make `const fn`s work with traits (in an untested
design, that is): qualify trait implementations and bounds with `const`.
This is necessary for meaningful interactions with operator overloading traits:
```rust
const fn map_vec3<T: Copy, F: const Fn(T) -> T>(xs: [T; 3], f: F) -> [T; 3] {
    [f([xs[0]), f([xs[1]), f([xs[2])]
}

const fn neg_vec3<T: Copy + const Neg>(xs: [T; 3]) -> [T; 3] {
    map_vec3(xs, |x| -x)
}

const impl Add for Point {
    fn add(self, other: Point) -> Point {
        Point {
            x: self.x + other.x,
            y: self.y + other.y
        }
    }
}
```
Having `const` trait methods (where all implementations are `const`) seems
useful, but it would not allow the usecase above on its own.
Trait implementations with `const` methods (instead of the entire `impl`
being `const`) would allow direct calls, but it's not obvious how one could
write a function generic over a type which implements a trait and requiring
that a certain method of that trait is implemented as `const`.

# Unresolved questions

* Keep recursion or disallow it for now? The conservative choice of having no
recursive `const fn`s would not affect the usecases intended for this RFC.
If we do allow it, we probably need a recursion limit, and/or an evaluation
algorithm that can handle *at least* tail recursion.
Also, there is no way to actually write a recursive `const fn` at this moment,
because no control flow primitives are implemented for constants, but that
cannot be taken for granted, at least `if`/`else` should eventually work.

# History

- This RFC was accepted on 2015-04-06. The primary concerns raised in
  the discussion concerned CTFE, and whether the `const fn` strategy
  locks us into an undesirable plan there.

# Updates since being accepted

Since it was accepted, the RFC has been updated as follows:

1. Allowed `const unsafe fn`
