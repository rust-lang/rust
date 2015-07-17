% Lang items

> **Note**: lang items are often provided by crates in the Rust distribution,
> and lang items themselves have an unstable interface. It is recommended to use
> officially distributed crates instead of defining your own lang items.

The `rustc` compiler has certain pluggable operations, that is,
functionality that isn't hard-coded into the language, but is
implemented in libraries, with a special marker to tell the compiler
it exists. The marker is the attribute `#[lang = "..."]` and there are
various different values of `...`, i.e. various different 'lang
items'.

For example, some traits in the standard library have special
treatment. One is `Copy`: there is a `copy` lang-item attached to the
`Copy` trait, and the Rust compiler treats this specially in two ways:

 1. If `x` has type that implements `Copy`, then `x` can be freely
    copied by the assignment operator:
    ```rust,ignore
    let y = x;
    let z = x;
    ```

 2. If a type tries to implement `Copy`, the Rust compiler will
    ensure that duplicating a value of that type will not cause any
    noncopyable type to be duplicated.

    For example, this code will be rejected:

    ```rust,ignore
    #[derive(Clone)]
    struct ThisTypeIsNotCopy(Box<i32>);

    #[derive(Clone)]
    struct TryToMakeThisCopy { okay: i32, not_okay: ThisTypeIsNotCopy }

    // This attempt to implement `Copy` will fail.
    impl Copy for TryToMakeThisCopy { }
    ```

The above two properties are both special qualities of the `Copy`
trait that other traits do not share, and thus they are associated
with whatever trait is given the `copy` lang item.

Here is a freestanding program that provides its own definition of the
`copy` lang item, that is slightly different than the definition in
the Rust standard library:

```
#![feature(lang_items, intrinsics, start, no_std)]
#![no_std]

#[lang = "copy"]
pub trait MyCopy {
    // Empty.
}

struct C(i32, i32);
impl MyCopy for C { }

#[start]
fn main(_argc: isize, _argv: *const *const u8) -> isize {
    let mut x = C(3, 4);
    let mut y = x;
    let mut z = x;
    x.0 = 5;
    y.0 = 6;
    z.0 = 7;

    #[link(name="c")]
    extern { fn printf(f: *const u8, ...); }

    let template = b"x: C(%d, %d) y: C(%d, %d), z: C(%d, %d)\n\0";
    unsafe { printf(template as *const u8, x.0, x.1, y.0, y.1, z.0, z.1); }
    return 0;
}

// Again, these functions and traits are used by the compiler, and are
// normally provided by libstd.

#[lang = "stack_exhausted"] extern fn stack_exhausted() {}
#[lang = "eh_personality"] extern fn eh_personality() {}
#[lang = "panic_fmt"] fn panic_fmt() -> ! { loop {} }

#[lang="sized"] pub trait Sized: PhantomFn {}
#[lang="phantom_fn"] pub trait PhantomFn {}
```

This compiles successfully. When we run the above code, it prints:
```text
x: C(5, 4) y: C(6, 4), z: C(7, 4)
```
So we can freely copy instances of `C`, since it implements `MyCopy`.

A potentially interesting detail about the above program is that it
differs from the Rust standard library in more than just the name
`MyCopy`. The `std::marker::Copy` extends `std::clone::Clone`,
ensuring that every type that implements `Copy` has a `clone` method.
The `MyCopy` trait does *not* extend `Clone`; these values have no
`clone` methods.

Other features provided by lang items include:

- overloadable operators via traits: the traits corresponding to the
  `==`, `<`, dereferencing (`*`) and `+` (etc.) operators are all
  marked with lang items; those specific four are `eq`, `ord`,
  `deref`, and `add` respectively.
- stack unwinding and general failure; the `eh_personality`, `panic`
  `panic_fmt`, and `panic_bounds_check` lang items.
- the traits in `std::marker` used to indicate types of
  various kinds; lang items `send`, `sync` and `copy`.
- the marker types and variance indicators found in
  `std::marker`; lang items `covariant_type`,
  `contravariant_lifetime`, etc.
- matching with string literal patterns; the `str_eq` lang item.

Lang items are loaded lazily by the compiler; e.g. if one never uses
array indexing (`a[i]`) then there is no need to define a function for
`panic_bounds_check`. `rustc` will emit an error when an item is
needed but not found in the current crate or any that it depends on.
