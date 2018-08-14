// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_snake_case)]

// Error messages for EXXXX errors.
// Each message should start and end with a new line, and be wrapped to 80 characters.
// In vim you can `:set tw=80` and use `gq` to wrap paragraphs. Use `:set tw=0` to disable.
register_long_diagnostics! {

E0178: r##"
In types, the `+` type operator has low precedence, so it is often necessary
to use parentheses.

For example:

```compile_fail,E0178
trait Foo {}

struct Bar<'a> {
    w: &'a Foo + Copy,   // error, use &'a (Foo + Copy)
    x: &'a Foo + 'a,     // error, use &'a (Foo + 'a)
    y: &'a mut Foo + 'a, // error, use &'a mut (Foo + 'a)
    z: fn() -> Foo + 'a, // error, use fn() -> (Foo + 'a)
}
```

More details can be found in [RFC 438].

[RFC 438]: https://github.com/rust-lang/rfcs/pull/438
"##,

E0536: r##"
The `not` cfg-predicate was malformed.

Erroneous code example:

```compile_fail,E0536
#[cfg(not())] // error: expected 1 cfg-pattern
pub fn something() {}

pub fn main() {}
```

The `not` predicate expects one cfg-pattern. Example:

```
#[cfg(not(target_os = "linux"))] // ok!
pub fn something() {}

pub fn main() {}
```

For more information about the cfg attribute, read:
https://doc.rust-lang.org/reference.html#conditional-compilation
"##,

E0537: r##"
An unknown predicate was used inside the `cfg` attribute.

Erroneous code example:

```compile_fail,E0537
#[cfg(unknown())] // error: invalid predicate `unknown`
pub fn something() {}

pub fn main() {}
```

The `cfg` attribute supports only three kinds of predicates:

 * any
 * all
 * not

Example:

```
#[cfg(not(target_os = "linux"))] // ok!
pub fn something() {}

pub fn main() {}
```

For more information about the cfg attribute, read:
https://doc.rust-lang.org/reference.html#conditional-compilation
"##,

E0538: r##"
Attribute contains same meta item more than once.

Erroneous code example:

```compile_fail,E0538
#[deprecated(
    since="1.0.0",
    note="First deprecation note.",
    note="Second deprecation note." // error: multiple same meta item
)]
fn deprecated_function() {}
```

Meta items are the key-value pairs inside of an attribute. Each key may only be
used once in each attribute.

To fix the problem, remove all but one of the meta items with the same key.

Example:

```
#[deprecated(
    since="1.0.0",
    note="First deprecation note."
)]
fn deprecated_function() {}
```
"##,

E0541: r##"
An unknown meta item was used.

Erroneous code example:

```compile_fail,E0541
#[deprecated(
    since="1.0.0",
    // error: unknown meta item
    reason="Example invalid meta item. Should be 'note'")
]
fn deprecated_function() {}
```

Meta items are the key-value pairs inside of an attribute. The keys provided
must be one of the valid keys for the specified attribute.

To fix the problem, either remove the unknown meta item, or rename it if you
provided the wrong name.

In the erroneous code example above, the wrong name was provided, so changing
to a correct one it will fix the error. Example:

```
#[deprecated(
    since="1.0.0",
    note="This is a valid meta item for the deprecated attribute."
)]
fn deprecated_function() {}
```
"##,

E0552: r##"
A unrecognized representation attribute was used.

Erroneous code example:

```compile_fail,E0552
#[repr(D)] // error: unrecognized representation hint
struct MyStruct {
    my_field: usize
}
```

You can use a `repr` attribute to tell the compiler how you want a struct or
enum to be laid out in memory.

Make sure you're using one of the supported options:

```
#[repr(C)] // ok!
struct MyStruct {
    my_field: usize
}
```

For more information about specifying representations, see the ["Alternative
Representations" section] of the Rustonomicon.

["Alternative Representations" section]: https://doc.rust-lang.org/nomicon/other-reprs.html
"##,

E0554: r##"
Feature attributes are only allowed on the nightly release channel. Stable or
beta compilers will not comply.

Example of erroneous code (on a stable compiler):

```ignore (depends on release channel)
#![feature(non_ascii_idents)] // error: #![feature] may not be used on the
                              //        stable release channel
```

If you need the feature, make sure to use a nightly release of the compiler
(but be warned that the feature may be removed or altered in the future).
"##,

E0557: r##"
A feature attribute named a feature that has been removed.

Erroneous code example:

```compile_fail,E0557
#![feature(managed_boxes)] // error: feature has been removed
```

Delete the offending feature attribute.
"##,

E0565: r##"
A literal was used in an attribute that doesn't support literals.

Erroneous code example:

```ignore (compile_fail not working here; see Issue #43707)
#![feature(attr_literals)]

#[inline("always")] // error: unsupported literal
pub fn something() {}
```

Literals in attributes are new and largely unsupported. Work to support literals
where appropriate is ongoing. Try using an unquoted name instead:

```
#[inline(always)]
pub fn something() {}
```
"##,

E0583: r##"
A file wasn't found for an out-of-line module.

Erroneous code example:

```ignore (compile_fail not working here; see Issue #43707)
mod file_that_doesnt_exist; // error: file not found for module

fn main() {}
```

Please be sure that a file corresponding to the module exists. If you
want to use a module named `file_that_doesnt_exist`, you need to have a file
named `file_that_doesnt_exist.rs` or `file_that_doesnt_exist/mod.rs` in the
same directory.
"##,

E0585: r##"
A documentation comment that doesn't document anything was found.

Erroneous code example:

```compile_fail,E0585
fn main() {
    // The following doc comment will fail:
    /// This is a useless doc comment!
}
```

Documentation comments need to be followed by items, including functions,
types, modules, etc. Examples:

```
/// I'm documenting the following struct:
struct Foo;

/// I'm documenting the following function:
fn foo() {}
```
"##,

E0586: r##"
An inclusive range was used with no end.

Erroneous code example:

```compile_fail,E0586
fn main() {
    let tmp = vec![0, 1, 2, 3, 4, 4, 3, 3, 2, 1];
    let x = &tmp[1..=]; // error: inclusive range was used with no end
}
```

An inclusive range needs an end in order to *include* it. If you just need a
start and no end, use a non-inclusive range (with `..`):

```
fn main() {
    let tmp = vec![0, 1, 2, 3, 4, 4, 3, 3, 2, 1];
    let x = &tmp[1..]; // ok!
}
```

Or put an end to your inclusive range:

```
fn main() {
    let tmp = vec![0, 1, 2, 3, 4, 4, 3, 3, 2, 1];
    let x = &tmp[1..=3]; // ok!
}
```
"##,

E0589: r##"
The value of `N` that was specified for `repr(align(N))` was not a power
of two, or was greater than 2^29.

```compile_fail,E0589
#[repr(align(15))] // error: invalid `repr(align)` attribute: not a power of two
enum Foo {
    Bar(u64),
}
```
"##,

E0658: r##"
An unstable feature was used.

Erroneous code example:

```compile_fail,E658
#[repr(u128)] // error: use of unstable library feature 'repr128'
enum Foo {
    Bar(u64),
}
```

If you're using a stable or a beta version of rustc, you won't be able to use
any unstable features. In order to do so, please switch to a nightly version of
rustc (by using rustup).

If you're using a nightly version of rustc, just add the corresponding feature
to be able to use it:

```
#![feature(repr128)]

#[repr(u128)] // ok!
enum Foo {
    Bar(u64),
}
```
"##,

E0633: r##"
The `unwind` attribute was malformed.

Erroneous code example:

```ignore (compile_fail not working here; see Issue #43707)
#[unwind()] // error: expected one argument
pub extern fn something() {}

fn main() {}
```

The `#[unwind]` attribute should be used as follows:

- `#[unwind(aborts)]` -- specifies that if a non-Rust ABI function
  should abort the process if it attempts to unwind. This is the safer
  and preferred option.

- `#[unwind(allowed)]` -- specifies that a non-Rust ABI function
  should be allowed to unwind. This can easily result in Undefined
  Behavior (UB), so be careful.

NB. The default behavior here is "allowed", but this is unspecified
and likely to change in the future.

"##,

E0705: r##"
A `#![feature]` attribute was declared for a feature that is stable in
the current edition.

Erroneous code example:

```ignore (limited to a warning during 2018 edition development)
#![feature(rust_2018_preview)]
#![feature(impl_header_lifetime_elision)] // error: the feature
                                          // `impl_header_lifetime_elision` is
                                          // included in the Rust 2018 edition
```

"##,

}

register_diagnostics! {
    E0539, // incorrect meta item
    E0540, // multiple rustc_deprecated attributes
    E0542, // missing 'since'
    E0543, // missing 'reason'
    E0544, // multiple stability levels
    E0545, // incorrect 'issue'
    E0546, // missing 'feature'
    E0547, // missing 'issue'
    E0548, // incorrect stability attribute type
    E0549, // rustc_deprecated attribute must be paired with either stable or unstable attribute
    E0550, // multiple deprecated attributes
    E0551, // incorrect meta item
    E0553, // multiple rustc_const_unstable attributes
    E0555, // malformed feature attribute, expected #![feature(...)]
    E0556, // malformed feature, expected just one word
    E0584, // file for module `..` found at both .. and ..
    E0629, // missing 'feature' (rustc_const_unstable)
    E0630, // rustc_const_unstable attribute must be paired with stable/unstable attribute
    E0693, // incorrect `repr(align)` attribute format
    E0694, // an unknown tool name found in scoped attributes
    E0703, // invalid ABI
    E0704, // incorrect visibility restriction
}
