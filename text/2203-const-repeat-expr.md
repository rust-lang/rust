- Feature Name: `const_repeat_expr`
- Start Date: 2017-10-20
- RFC PR: [rust-lang/rfcs#2203](https://github.com/rust-lang/rfcs/pull/2203)
- Rust Issue: [rust-lang/rust#49147](https://github.com/rust-lang/rust/issues/49147)

# Summary
[summary]: #summary

Relaxes the rules for repeat expressions, `[x; N]` such that `x` may also be
`const` *(strictly speaking rvalue promotable)*, in addition to `typeof(x): Copy`.
The result of `[x; N]` where `x` is `const` is itself also `const`.

# Motivation
[motivation]: #motivation

[RFC 2000, `const_generics`]: https://github.com/rust-lang/rfcs/blob/master/text/2000-const-generics.md
[`const_default` RFC]: https://github.com/Centril/rfcs/blob/rfc/const-default/text/0000-const-default.md

[RFC 2000, `const_generics`] introduced the ability to have generically sized
arrays. Even with that RFC, it is currently impossible to create such an array
that is also `const`. Creating an array that is `const` may for example be
useful for the [`const_default` RFC] which proposes the following trait:

```rust
pub trait ConstDefault { const DEFAULT: Self; }
```

To add an implementation of this trait for an array of any size where the
elements of type `T` are `ConstDefault`, as in:

```rust
impl<T: ConstDefault, const N: usize> ConstDefault for [T; N] {
    const DEFAULT: Self = [T::DEFAULT; N];
}
```

[`mem::uninitialized()`]: https://doc.rust-lang.org/nightly/std/mem/fn.uninitialized.html

In the example given by [`mem::uninitialized()`], a value of type
`[Vec<u32>; 1000]` is created and filled. With this RFC, and when `Vec::new()`
becomes const, the user can simply write:

```rust
let data = [Vec::<u32>::new(); 1000];
println!("{:?}", &data[0]);
```

this removes one common reason to use `uninitialized()` which **"is incredibly
dangerous"**.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

You have a variable or expression `X` which is const, for example:

```rust
type T = Option<Box<u32>>;
const X: T = None;
```

Now, you'd like to use array repeat expressions `[X; N]` to create an array
containing a bunch of `X`es. Sorry, you are out of luck!

But with this RFC, you can now write:

```rust
const X: T = None;
const arr: [T; 100] = [X; 100];
```

or, if you wish to modify the array later:

```rust
const X: T = None;
let mut arr = [X; 100];
arr[0] = Some(Box::new(1));
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

Values which are `const` are freely duplicatable as seen in the following
example which compiles today. This is also the case with `Copy`. Therefore, the
value `X` in the repeat expression may be simply treated as if it were of a
`Copy` type.

```rust
fn main() {
    type T = Option<Box<u32>>;
    const X: T = None;
    let mut arr = [X, X];
    arr[0] = Some(Box::new(1));
}
```

Thus, the compiler may rewrite the following:

```rust
fn main() {
    type T = Option<Box<u32>>;
    const X: T = None;
    let mut arr = [X; 2];
    arr[0] = Some(Box::new(1));
}
```

internally as:

```rust
fn main() {
    type T = Option<Box<u32>>;

    // This is the value to be repeated.
    // In this case, a panic won't happen, but if it did, that panic
    // would happen during compile time at this point and not later.
    const X: T = None;

    let mut arr = {
        let mut data: [T; 2];

        unsafe {
            data = mem::uninitialized();

            let mut iter = (&mut data[..]).into_iter();
            while let Some(elem) = iter.next() {
                // ptr::write does not run destructor of elem already in array.
                // Since X is const, it can not panic at this point.
                ptr::write(elem, X);
            }
        }

        data
    };

    arr[0] = Some(Box::new(1));
}
```

Additionally, the pass that checks `const`ness must treat `[expr; N]` as a
`const` value such that `[expr; N]` is assignable to a `const` item as well
as permitted inside a `const fn`.

Strictly speaking, the set of values permitted in the expression `[expr; N]`
are those where `is_rvalue_promotable(expr)` or `typeof(expr): Copy`.
Specifically, in `[expr; N]` the expression `expr` is evaluated:
+ never, if `N == 0`,
+ one time, if `N == 1`,
+ `N` times, otherwise.

For values that are not freely duplicatable, evaluating `expr` will result in
a move, which results in an error if `expr` is moved more than once (including
moves outside of the repeat expression). These semantics are intentionally
conservative and intended to be forward-compatible with a more expansive
`is_const(expr)` check.

# Drawbacks
[drawbacks]: #drawbacks

It might make the semantics of array initializers more fuzzy. The RFC, however,
argues that the change is quite intuitive.

# Rationale and alternatives
[alternatives]: #alternatives

[`ptr::write(..)`]: https://doc.rust-lang.org/nightly/std/ptr/fn.write.html

The alternative, in addition to simply not doing this, is to modify a host of
other constructs such as [`mem::uninitialized()`], for loops over iterators,
[`ptr::write`] to be `const`, which is is a larger change. The design offered by
this RFC is therefore the simplest and most non-intrusive design. It is also
the most consistent.

Another alternative is to allow a more expansive set of values `is_const(expr)`
rather than `is_rvalue_promotable(expr)`. A consequence of this is that checking
constness would be done earlier on the HIR. Instead, checking if `expr` is
rvalue promotable can be done on the MIR and does not require significant
changes to the compiler. If we decide to expand to `is_const(expr)` in the
future, we may still do so as the changes proposed in this RFC are
compatible with such future changes.

The impact of not doing this change is to not enable generically sized arrays to
be `const` as well as encouraging the use of `mem::uninitialized`.

# Unresolved questions
[unresolved]: #unresolved-questions

There are no unresolved questions.
