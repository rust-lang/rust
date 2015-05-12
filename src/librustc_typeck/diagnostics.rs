// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_snake_case)]

register_long_diagnostics! {

E0046: r##"
When trying to make some type implement a trait `Foo`, you must, at minimum,
provide implementations for all of `Foo`'s required methods (meaning the
methods that do not have default implementations), as well as any required
trait items like associated types or constants.
"##,

E0054: r##"
It is not allowed to cast to a bool. If you are trying to cast a numeric type
to a bool, you can compare it with zero instead:

```
let x = 5;

// Ok
let x_is_nonzero = x != 0;

// Not allowed, won't compile
let x_is_nonzero = x as bool;
```
"##,

E0062: r##"
This error indicates that during an attempt to build a struct or struct-like
enum variant, one of the fields was specified more than once. Each field should
be specified exactly one time.
"##,

E0063: r##"
This error indicates that during an attempt to build a struct or struct-like
enum variant, one of the fields was not provided. Each field should be specified
exactly once.
"##,

E0067: r##"
The left-hand side of an assignment operator must be an lvalue expression. An
lvalue expression represents a memory location and includes item paths (ie,
namespaced variables), dereferences, indexing expressions, and field references.

```
use std::collections::LinkedList;

// Good
let mut list = LinkedList::new();


// Bad: assignment to non-lvalue expression
LinkedList::new() += 1;
```
"##,

E0081: r##"
Enum discriminants are used to differentiate enum variants stored in memory.
This error indicates that the same value was used for two or more variants,
making them impossible to tell apart.

```
// Good.
enum Enum {
    P,
    X = 3,
    Y = 5
}

// Bad.
enum Enum {
    P = 3,
    X = 3,
    Y = 5
}
```

Note that variants without a manually specified discriminant are numbered from
top to bottom starting from 0, so clashes can occur with seemingly unrelated
variants.

```
enum Bad {
    X,
    Y = 0
}
```

Here `X` will have already been assigned the discriminant 0 by the time `Y` is
encountered, so a conflict occurs.
"##,

E0082: r##"
The default type for enum discriminants is `isize`, but it can be adjusted by
adding the `repr` attribute to the enum declaration. This error indicates that
an integer literal given as a discriminant is not a member of the discriminant
type. For example:

```
#[repr(u8)]
enum Thing {
    A = 1024,
    B = 5
}
```

Here, 1024 lies outside the valid range for `u8`, so the discriminant for `A` is
invalid. You may want to change representation types to fix this, or else change
invalid discriminant values so that they fit within the existing type.

Note also that without a representation manually defined, the compiler will
optimize by using the smallest integer type possible.
"##,

E0083: r##"
At present, it's not possible to define a custom representation for an enum with
a single variant. As a workaround you can add a `Dummy` variant.

See: https://github.com/rust-lang/rust/issues/10292
"##,

E0084: r##"
It is impossible to define an integer type to be used to represent zero-variant
enum values because there are no zero-variant enum values. There is no way to
construct an instance of the following type using only safe code:

```
enum Empty {}
```
"##,

E0131: r##"
It is not possible to define `main` with type parameters, or even with function
parameters. When `main` is present, it must take no arguments and return `()`.
"##,

E0132: r##"
It is not possible to declare type parameters on a function that has the `start`
attribute. Such a function must have the following type signature:

```
fn(isize, *const *const u8) -> isize
```
"##,

E0184: r##"
Explicitly implementing both Drop and Copy for a type is currently disallowed.
This feature can make some sense in theory, but the current implementation is
incorrect and can lead to memory unsafety (see [issue #20126][iss20126]), so
it has been disabled for now.

[iss20126]: https://github.com/rust-lang/rust/issues/20126
"##,

E0204: r##"
An attempt to implement the `Copy` trait for a struct failed because one of the
fields does not implement `Copy`. To fix this, you must implement `Copy` for the
mentioned field. Note that this may not be possible, as in the example of

```
struct Foo {
    foo : Vec<u32>,
}

impl Copy for Foo { }
```

This fails because `Vec<T>` does not implement `Copy` for any `T`.

Here's another example that will fail:

```
#[derive(Copy)]
struct Foo<'a> {
    ty: &'a mut bool,
}
```

This fails because `&mut T` is not `Copy`, even when `T` is `Copy` (this
differs from the behavior for `&T`, which is `Copy` when `T` is `Copy`).
"##,

E0205: r##"
An attempt to implement the `Copy` trait for an enum failed because one of the
variants does not implement `Copy`. To fix this, you must implement `Copy` for
the mentioned variant. Note that this may not be possible, as in the example of

```
enum Foo {
    Bar(Vec<u32>),
    Baz,
}

impl Copy for Foo { }
```

This fails because `Vec<T>` does not implement `Copy` for any `T`.

Here's another example that will fail:

```
#[derive(Copy)]
enum Foo<'a> {
    Bar(&'a mut bool),
    Baz
}
```

This fails because `&mut T` is not `Copy`, even when `T` is `Copy` (this
differs from the behavior for `&T`, which is `Copy` when `T` is `Copy`).
"##,

E0206: r##"
You can only implement `Copy` for a struct or enum. Both of the following
examples will fail, because neither `i32` (primitive type) nor `&'static Bar`
(reference to `Bar`) is a struct or enum:

```
type Foo = i32;
impl Copy for Foo { } // error

#[derive(Copy, Clone)]
struct Bar;
impl Copy for &'static Bar { } // error
```
"##,

E0243: r##"
This error indicates that not enough type parameters were found in a type or
trait.

For example, the `Foo` struct below is defined to be generic in `T`, but the
type parameter is missing in the definition of `Bar`:

```
struct Foo<T> { x: T }

struct Bar { x: Foo }
```
"##,

E0244: r##"
This error indicates that too many type parameters were found in a type or
trait.

For example, the `Foo` struct below has no type parameters, but is supplied
with two in the definition of `Bar`:

```
struct Foo { x: bool }

struct Bar<S, T> { x: Foo<S, T> }
```
"##,

E0249: r##"
This error indicates a constant expression for the array length was found, but
it was not an integer (signed or unsigned) expression.

Some examples of code that produces this error are:

```
const A: [u32; "hello"] = []; // error
const B: [u32; true] = []; // error
const C: [u32; 0.0] = []; // error
"##,

E0250: r##"
This means there was an error while evaluating the expression for the length of
a fixed-size array type.

Some examples of code that produces this error are:

```
// divide by zero in the length expression
const A: [u32; 1/0] = [];

// Rust currently will not evaluate the function `foo` at compile time
fn foo() -> usize { 12 }
const B: [u32; foo()] = [];

// it is an error to try to add `u8` and `f64`
use std::{f64, u8};
const C: [u32; u8::MAX + f64::EPSILON] = [];
```
"##

}

register_diagnostics! {
    E0023,
    E0024,
    E0025,
    E0026,
    E0027,
    E0029,
    E0030,
    E0031,
    E0033,
    E0034, // multiple applicable methods in scope
    E0035, // does not take type parameters
    E0036, // incorrect number of type parameters given for this method
    E0038, // cannot convert to a trait object because trait is not object-safe
    E0040, // explicit use of destructor method
    E0044, // foreign items may not have type parameters
    E0045, // variadic function must have C calling convention
    E0049,
    E0050,
    E0053,
    E0055, // method has an incompatible type for trait
    E0057, // method has an incompatible type for trait
    E0059,
    E0060,
    E0061,
    E0066,
    E0068,
    E0069,
    E0070,
    E0071,
    E0072,
    E0073,
    E0074,
    E0075,
    E0076,
    E0077,
    E0085,
    E0086,
    E0087,
    E0088,
    E0089,
    E0090,
    E0091,
    E0092,
    E0093,
    E0094,
    E0101,
    E0102,
    E0103,
    E0104,
    E0106,
    E0107,
    E0116,
    E0117,
    E0118,
    E0119,
    E0120,
    E0121,
    E0122,
    E0123,
    E0124,
    E0127,
    E0128,
    E0129,
    E0130,
    E0141,
    E0159,
    E0163,
    E0164,
    E0166,
    E0167,
    E0168,
    E0172,
    E0173, // manual implementations of unboxed closure traits are experimental
    E0174, // explicit use of unboxed closure methods are experimental
    E0178,
    E0182,
    E0183,
    E0185,
    E0186,
    E0187, // can't infer the kind of the closure
    E0188, // types differ in mutability
    E0189, // can only cast a boxed pointer to a boxed object
    E0190, // can only cast a &-pointer to an &-object
    E0191, // value of the associated type must be specified
    E0192, // negative imples are allowed just for `Send` and `Sync`
    E0193, // cannot bound type where clause bounds may only be attached to types
           // involving type parameters
    E0194,
    E0195, // lifetime parameters or bounds on method do not match the trait declaration
    E0196, // cannot determine a type for this closure
    E0197, // inherent impls cannot be declared as unsafe
    E0198, // negative implementations are not unsafe
    E0199, // implementing trait is not unsafe
    E0200, // trait requires an `unsafe impl` declaration
    E0201, // duplicate method in trait impl
    E0202, // associated items are not allowed in inherent impls
    E0203, // type parameter has more than one relaxed default bound,
           // and only one is supported
    E0207, // type parameter is not constrained by the impl trait, self type, or predicate
    E0208,
    E0209, // builtin traits can only be implemented on structs or enums
    E0210, // type parameter is not constrained by any local type
    E0211,
    E0212, // cannot extract an associated type from a higher-ranked trait bound
    E0213, // associated types are not accepted in this context
    E0214, // parenthesized parameters may only be used with a trait
    E0215, // angle-bracket notation is not stable with `Fn`
    E0216, // parenthetical notation is only stable with `Fn`
    E0217, // ambiguous associated type, defined in multiple supertraits
    E0218, // no associated type defined
    E0219, // associated type defined in higher-ranked supertrait
    E0220, // associated type not found for type parameter
    E0221, // ambiguous associated type in bounds
    E0222, // variadic function must have C calling convention
    E0223, // ambiguous associated type
    E0224, // at least one non-builtin train is required for an object type
    E0225, // only the builtin traits can be used as closure or object bounds
    E0226, // only a single explicit lifetime bound is permitted
    E0227, // ambiguous lifetime bound, explicit lifetime bound required
    E0228, // explicit lifetime bound required
    E0229, // associated type bindings are not allowed here
    E0230, // there is no type parameter on trait
    E0231, // only named substitution parameters are allowed
    E0232, // this attribute must have a value
    E0233,
    E0234, // `for` loop expression has type which does not implement the `Iterator` trait
    E0235, // structure constructor specifies a structure of type but
    E0236, // no lang item for range syntax
    E0237, // no lang item for range syntax
    E0238, // parenthesized parameters may only be used with a trait
    E0239, // `next` method of `Iterator` trait has unexpected type
    E0240,
    E0241,
    E0242, // internal error looking up a definition
    E0245, // not a trait
    E0246, // illegal recursive type
    E0247, // found module name used as a type
    E0248, // found value name used as a type
    E0318, // can't create default impls for traits outside their crates
    E0319, // trait impls for defaulted traits allowed just for structs/enums
    E0320, // recursive overflow during dropck
    E0321, // extended coherence rules for defaulted traits violated
    E0322, // cannot implement Sized explicitly
    E0323, // implemented an associated const when another trait item expected
    E0324, // implemented a method when another trait item expected
    E0325, // implemented an associated type when another trait item expected
    E0326, // associated const implemented with different type from trait
    E0327, // referred to method instead of constant in match pattern
    E0366, // dropck forbid specialization to concrete type or region
    E0367, // dropck forbid specialization to predicate not in struct/enum
    E0368, // binary operation `<op>=` cannot be applied to types
    E0369, // binary operation `<op>` cannot be applied to types
    E0371, // impl Trait for Trait is illegal
    E0372  // impl Trait for Trait where Trait is not object safe
}
