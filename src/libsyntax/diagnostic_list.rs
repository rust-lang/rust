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

E0534: r##"
The `inline` attribute was malformed.

Erroneous code example:

```compile_fail,E0534
#[inline()] // error: expected one argument
pub fn something() {}

fn main() {}
```

The parenthesized `inline` attribute requires the parameter to be specified:

```ignore
#[inline(always)]
fn something() {}

// or:

#[inline(never)]
fn something() {}
```

Alternatively, a paren-less version of the attribute may be used to hint the
compiler about inlining opportunity:

```
#[inline]
fn something() {}
```

For more information about the inline attribute, read:
https://doc.rust-lang.org/reference.html#inline-attributes
"##,

E0535: r##"
An unknown argument was given to the `inline` attribute.

Erroneous code example:

```compile_fail,E0535
#[inline(unknown)] // error: invalid argument
pub fn something() {}

fn main() {}
```

The `inline` attribute only supports two arguments:

 * always
 * never

All other arguments given to the `inline` attribute will return this error.
Example:

```
#[inline(never)] // ok!
pub fn something() {}

fn main() {}
```

For more information about the inline attribute, https:
read://doc.rust-lang.org/reference.html#inline-attributes
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

E0558: r##"
The `export_name` attribute was malformed.

Erroneous code example:

```compile_fail,E0558
#[export_name] // error: export_name attribute has invalid format
pub fn something() {}

fn main() {}
```

The `export_name` attribute expects a string in order to determine the name of
the exported symbol. Example:

```
#[export_name = "some_function"] // ok!
pub fn something() {}

fn main() {}
```
"##,

E0565: r##"
A literal was used in an attribute that doesn't support literals.

Erroneous code example:

```compile_fail,E0565
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
}

register_diagnostics! {
    E0538, // multiple [same] items
    E0539, // incorrect meta item
    E0540, // multiple rustc_deprecated attributes
    E0541, // unknown meta item
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
    E0552, // unrecognized representation hint
    E0553, // unrecognized enum representation hint
    E0554, // #[feature] may not be used on the [] release channel
    E0555, // malformed feature attribute, expected #![feature(...)]
    E0556, // malformed feature, expected just one word
    E0557, // feature has been removed
}
