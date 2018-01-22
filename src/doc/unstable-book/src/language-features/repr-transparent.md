# `repr_transparent`

The tracking issue for this feature is: [#43036]

[#43036]: https://github.com/rust-lang/rust/issues/43036

------------------------

This feature enables the `repr(transparent)` attribute on structs, which enables
the use of newtypes without the usual ABI implications of wrapping the value in
a struct.

## Background

It's sometimes useful to add additional type safety by introducing *newtypes*.
For example, code that handles numeric quantities in different units such as
millimeters, centimeters, grams, kilograms, etc. may want to use the type system
to rule out mistakes such as adding millimeters to grams:

```rust
use std::ops::Add;

struct Millimeters(f64);
struct Grams(f64);

impl Add<Millimeters> for Millimeters {
    type Output = Millimeters;

    fn add(self, other: Millimeters) -> Millimeters {
        Millimeters(self.0 + other.0)
    }
}

// Likewise: impl Add<Grams> for Grams {}
```

Other uses of newtypes include using `PhantomData` to add lifetimes to raw
pointers or to implement the "phantom types" pattern. See the [PhantomData]
documentation and [the Nomicon][nomicon-phantom] for more details.

The added type safety is especially useful when interacting with C or other
languages. However, in those cases we need to ensure the newtypes we add do not
introduce incompatibilities with the C ABI.

## Newtypes in FFI

Luckily, `repr(C)` newtypes are laid out just like the type they wrap on all
platforms which Rust currently supports, and likely on many more. For example,
consider this C declaration:

```C
struct Object {
    double weight; //< in grams
    double height; //< in millimeters
    // ...
}

void frobnicate(struct Object *);
```

While using this C code from Rust, we could add `repr(C)` to the `Grams` and
`Millimeters` newtypes introduced above and use them to add some type safety
while staying compatible with the memory layout of `Object`:

```rust,no_run
#[repr(C)]
struct Grams(f64);

#[repr(C)]
struct Millimeters(f64);

#[repr(C)]
struct Object {
    weight: Grams,
    height: Millimeters,
    // ...
}

extern {
    fn frobnicate(_: *mut Object);
}
```

This works even when adding some `PhantomData` fields, because they are
zero-sized and therefore don't have to affect the memory layout.

However, there's more to the ABI than just memory layout: there's also the
question of how function call arguments and return values are passed. Many
common ABI treat a struct containing a single field differently from that field
itself, at least when the field is a scalar (e.g., integer or float or pointer).

To continue the above example, suppose the C library also exposes a function
like this:

```C
double calculate_weight(double height);
```

Using our newtypes on the Rust side like this will cause an ABI mismatch on many
platforms:

```rust,ignore
extern {
    fn calculate_weight(height: Millimeters) -> Grams;
}
```

For example, on x86_64 Linux, Rust will pass the argument in an integer
register, while the C function expects the argument to be in a floating-point
register. Likewise, the C function will return the result in a floating-point
register while Rust will expect it in an integer register.

Note that this problem is not specific to floats: To give another example,
32-bit x86 linux will pass and return `struct Foo(i32);` on the stack while
`i32` is placed in registers.

## Enter `repr(transparent)`

So while `repr(C)` happens to do the right thing with respect to memory layout,
it's not quite the right tool for newtypes in FFI. Instead of declaring a C
struct, we need to communicate to the Rust compiler that our newtype is just for
type safety on the Rust side. This is what `repr(transparent)` does.

The attribute can be applied to a newtype-like structs that contains a single
field. It indicates that the newtype should be represented exactly like that
field's type, i.e., the newtype should be ignored for ABI purpopses: not only is
it laid out the same in memory, it is also passed identically in function calls.

In the above example, the ABI mismatches can be prevented by making the newtypes
`Grams` and `Millimeters` transparent like this:

```rust
#![feature(repr_transparent)]

#[repr(transparent)]
struct Grams(f64);

#[repr(transparent)]
struct Millimeters(f64);
```

In addition to that single field, any number of zero-sized fields are permitted,
including but not limited to `PhantomData`:

```rust
#![feature(repr_transparent)]

use std::marker::PhantomData;

struct Foo { /* ... */ }

#[repr(transparent)]
struct FooPtrWithLifetime<'a>(*const Foo, PhantomData<&'a Foo>);

#[repr(transparent)]
struct NumberWithUnit<T, U>(T, PhantomData<U>);

struct CustomZst;

#[repr(transparent)]
struct PtrWithCustomZst<'a> {
    ptr: FooPtrWithLifetime<'a>,
    some_marker: CustomZst,
}
```

Transparent structs can be nested: `PtrWithCustomZst` is also represented
exactly like `*const Foo`.

Because `repr(transparent)` delegates all representation concerns to another
type, it is incompatible with all other `repr(..)` attributes. It also cannot be
applied to enums, unions, empty structs, structs whose fields are all
zero-sized, or structs with *multiple* non-zero-sized fields.

[PhantomData]: https://doc.rust-lang.org/std/marker/struct.PhantomData.html
[nomicon-phantom]: https://doc.rust-lang.org/nomicon/phantom-data.html
