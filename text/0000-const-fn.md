- Feature Name: const_fn
- Start Date: 2015-02-25
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

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
filled in by a macro - and they're marked "stable".

A pre-RFC for the removal of the dangerous (and oftenly misued) `static mut`
received positive feedback, but only under the condition that abstractions
could be created and used in `const` and `static` items.

Another concern is the ability to use certain intrinsics, like `size_of`, inside
constant expressions, including fixed-length array types. Unlike keyword-based
alternatives, `const fn` provides an extensible and composable building block
for such features.

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
Only simple by-value immutable bindings are allowed as arguments' patterns.
The body of the function is checked as if it were a block inside a `const`:
```rust
const FOO: Foo = {
    // Currently, only item "statements" are allowed here.
    stmts;
    // The function's arguments and constant expressions can be freely combined.
    expr
}
```
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

# Drawbacks

None that I know of.

# Alternatives

* Not do anything for 1.0. This would result in some APIs being crippled and
serious backwards compatibility issues - `UnsafeCell`'s `value` field cannot
simply be removed later.
* While not an alternative, but rather a potential extension, there is only way
I could make `const fn`s work with traits (in an untested design, that is):
qualify trait implementations and bounds with `const`. This is necessary for
meaningful interactions with overloading traits - quick example:
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
useful, but is not enough of its own.

# Unresolved questions

Should we allow `unsafe const fn`? The implementation cost is neglible, but I
am not certain it needs to exist.
