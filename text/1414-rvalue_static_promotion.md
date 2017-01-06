- Feature Name: rvalue_static_promotion
- Start Date: 2015-12-18
- RFC PR: [#1414](https://github.com/rust-lang/rfcs/pull/1414)
- Rust Issue: [#38865](https://github.com/rust-lang/rust/issues/38865)

# Summary
[summary]: #summary

Promote constexpr rvalues to values in static memory instead of
stack slots, and expose those in the language by being able to directly create
`'static` references to them. This would allow code like
`let x: &'static u32 = &42` to work.

# Motivation
[motivation]: #motivation

Right now, when dealing with constant values, you have to explicitly define
`const` or `static` items to create references with `'static` lifetime,
which can be unnecessarily verbose if those items never get exposed
in the actual API:

```rust
fn return_x_or_a_default(x: Option<&u32>) -> &u32 {
    if let Some(x) = x {
        x
    } else {
        static DEFAULT_X: u32 = 42;
        &DEFAULT_X
    }
}
fn return_binop() -> &'static Fn(u32, u32) -> u32 {
    const STATIC_TRAIT_OBJECT: &'static Fn(u32, u32) -> u32
        = &|x, y| x + y;
    STATIC_TRAIT_OBJECT
}
```

This workaround also has the limitation of not being able to refer to
type parameters of a containing generic functions, eg you can't do this:

```rust
fn generic<T>() -> &'static Option<T> {
    const X: &'static Option<T> = &None::<T>;
    X
}
```

However, the compiler already special cases a small subset of rvalue
const expressions to have static lifetime - namely the empty array expression:

```rust
let x: &'static [u8] = &[];
```

And though they don't have to be seen as such, string literals could be regarded
as the same kind of special sugar:

```rust
let b: &'static [u8; 4] = b"test";
// could be seen as `= &[116, 101, 115, 116]`

let s: &'static str = "foo";
// could be seen as `= &str([102, 111, 111])`
// given `struct str([u8]);` and the ability to construct compound
// DST structs directly
```

With the proposed change, those special cases would instead become
part of a general language feature usable for custom code.

# Detailed design
[design]: #detailed-design

Inside a function body's block:

- If a shared reference to a constexpr rvalue is taken. (`&<constexpr>`)
- And the constexpr does not contain a `UnsafeCell { ... }` constructor.
- And the constexpr does not contain a const fn call returning a type containing a `UnsafeCell`.
- Then instead of translating the value into a stack slot, translate
  it into a static memory location and give the resulting reference a
  `'static` lifetime.

The `UnsafeCell` restrictions are there to ensure that the promoted value is
truly immutable behind the reference.

Examples:

```rust
// OK:
let a: &'static u32 = &32;
let b: &'static Option<UnsafeCell<u32>> = &None;
let c: &'static Fn() -> u32 = &|| 42;

let h: &'static u32 = &(32 + 64);

fn generic<T>() -> &'static Option<T> {
    &None::<T>
}

// BAD:
let f: &'static Option<UnsafeCell<u32>> = &Some(UnsafeCell { data: 32 });
let g: &'static Cell<u32> = &Cell::new(); // assuming conf fn new()
```

These rules above should be consistent with the existing rvalue promotions in `const`
initializer expressions:

```rust
// If this compiles:
const X: &'static T = &<constexpr foo>;

// Then this should compile as well:
let x: &'static T = &<constexpr foo>;
```

## Implementation

The necessary changes in the compiler did already get implemented as
part of codegen optimizations (emitting references-to or memcopies-from values in static memory instead of embedding them in the code).

All that is left do do is "throw the switch" for the new lifetime semantic
by removing these lines:
https://github.com/rust-lang/rust/blob/29ea4eef9fa6e36f40bc1f31eb1e56bf5941ee72/src/librustc/middle/mem_categorization.rs#L801-L807

(And of course fixing any fallout/bitrot that might have happened, adding tests, etc.)

# Drawbacks
[drawbacks]: #drawbacks

One more feature with seemingly ad-hoc rules to complicate the language...

# Alternatives, Extensions
[alternatives]: #alternatives

It would be possible to extend support to `&'static mut` references,
as long as there is the additional constraint that the
referenced type is zero sized.

This again has precedence in the array reference constructor:

```rust
// valid code today
let y: &'static mut [u8] = &mut [];
```

The rules would be similar:

- If a mutable reference to a constexpr rvalue is taken. (`&mut <constexpr>`)
- And the constexpr does not contain a `UnsafeCell { ... }` constructor.
- And the constexpr does not contain a const fn call returning a type containing a `UnsafeCell`.
- _And the type of the rvalue is zero-sized._
- Then instead of translating the value into a stack slot, translate
  it into a static memory location and give the resulting reference a
  `'static` lifetime.

The zero-sized restriction is there because
aliasing mutable references are only safe for zero sized types
(since you never dereference the pointer for them).

Example:

```rust
fn return_fn_mut_or_default(&mut self) -> &FnMut(u32, u32) -> u32 {
    self.operator.unwrap_or(&mut |x, y| x * y)
    // ^ would be okay, since it would be translated like this:
    // const STATIC_TRAIT_OBJECT: &'static mut FnMut(u32, u32) -> u32
    //     = &mut |x, y| x * y;
    // self.operator.unwrap_or(STATIC_TRAIT_OBJECT)
}

let d: &'static mut () = &mut ();
let e: &'static mut Fn() -> u32 = &mut || 42;
```

There are two ways this could be taken further with zero-sized types:

1. Remove the `UnsafeCell` restriction if the type of the rvalue is zero-sized.
2. The above, but also remove the __constexpr__ restriction, applying to any zero-sized rvalue instead.

Both cases would work because one can't cause memory unsafety with a reference
to a zero sized value, and they would allow more safe code to compile.

However, they might complicated reasoning about the rules more,
especially with the last one also being possibly confusing in regards to
side-effects.

Not doing this means:

- Relying on `static` and `const` items to create `'static` references, which won't work in generics.
- Empty-array expressions would remain special cased.
- It would also not be possible to safely create `&'static mut` references to zero-sized
types, though that part could also be achieved by allowing mutable references to
zero-sized types in constants.

# Unresolved questions
[unresolved]: #unresolved-questions

None, beyond "Should we do alternative 1 instead?".
