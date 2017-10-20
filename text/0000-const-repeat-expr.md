- Feature Name: const_repeat_expr
- Start Date: 2017-10-20
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Relaxes the rules for repeat expressions, `[x; N]` such that `x` may also be
`const`, in addition to `typeof(x): Copy`. The result of `[x; N]` where `x` is
`const` is itself also `const`.

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

[`std::mem::uninitialized()`]: https://doc.rust-lang.org/nightly/std/mem/fn.uninitialized.html

In the example given by [`std::mem::uninitialized()`], a value of type
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
const arr: [u32; 100] = [X; 100];
```

or, if you wish to modify the array later:

```rust
const X: T = None;
let mut arr = [x; 100];
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

Thus, the compiler may rewrite this internally as:

```rust
// This is the value to be repeated and typeof(X) the type it has.
const X: typeof(X) = VALUE;
// N is the size of the array and how many times to repeat X.
const N: usize = SIZE;

{
    let mut data: [typeof(X); N];

    unsafe {
        data = mem::uninitialized();
    
        let mut iter = (&mut data[..]).into_iter();
        while let Some(elem) = iter.next() {
            // ptr::write does not run destructor of elem already in array.
            // Since X is const, it can not panic.
            ptr::write(elem, X);
        }
    }

    data
}
```

Additionally, the pass that checks `const`ness must treat `[X; N]` as a `const`
value.

# Drawbacks
[drawbacks]: #drawbacks

It might make the semantics of array initializers more fuzzy. The RFC, however,
argues that the change is quite intuitive.

# Rationale and alternatives
[alternatives]: #alternatives

[`ptr::write(..)`]: https://doc.rust-lang.org/nightly/std/ptr/fn.write.html

The alternative, in addition to simply not doing this, is to modify a host of
other constructs such as [`std::mem::uninitialized()`], for loops over iterators,
[`ptr::write`] to be `const`, which is is a larger change. The design offered by
this RFC is therefore the simplest and most non-intrusive design. It is also
the most consistent.

The impact of not doing this change is to not enable generically sized arrays to
be `const`.

# Unresolved questions
[unresolved]: #unresolved-questions

[`drop_types_in_const`]: https://github.com/rust-lang/rfcs/blob/master/text/1440-drop-types-in-const.md

The relation to [`drop_types_in_const`] should be resolved during the RFC process.
The soundness of the proposal should be verified.
Other than that, there are no unresolved questions.