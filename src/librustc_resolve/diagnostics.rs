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

// Error messages for EXXXX errors.
// Each message should start and end with a new line, and be wrapped to 80 characters.
// In vim you can `:set tw=80` and use `gq` to wrap paragraphs. Use `:set tw=0` to disable.
register_long_diagnostics! {

E0154: r##"
Imports (`use` statements) are not allowed after non-item statements, such as
variable declarations and expression statements.

Here is an example that demonstrates the error:
```
fn f() {
    // Variable declaration before import
    let x = 0;
    use std::io::Read;
    ...
}
```

The solution is to declare the imports at the top of the block, function, or
file.

Here is the previous example again, with the correct order:
```
fn f() {
    use std::io::Read;
    let x = 0;
    ...
}
```

See the Declaration Statements section of the reference for more information
about what constitutes an Item declaration and what does not:

http://doc.rust-lang.org/reference.html#statements
"##,

E0259: r##"
The name chosen for an external crate conflicts with another external crate that
has been imported into the current module.

Wrong example:
```
extern crate a;
extern crate crate_a as a;
```

The solution is to choose a different name that doesn't conflict with any
external crate imported into the current module.

Correct example:
```
extern crate a;
extern crate crate_a as other_name;
```
"##,

E0260: r##"
The name for an item declaration conflicts with an external crate's name.

For instance,
```
extern crate abc;

struct abc;
```

There are two possible solutions:

Solution #1: Rename the item.

```
extern crate abc;

struct xyz;
```

Solution #2: Import the crate with a different name.

```
extern crate abc as xyz;

struct abc;
```

See the Declaration Statements section of the reference for more information
about what constitutes an Item declaration and what does not:

http://doc.rust-lang.org/reference.html#statements
"##,

E0317: r##"
User-defined types or type parameters cannot shadow the primitive types.
This error indicates you tried to define a type, struct or enum with the same
name as an existing primitive type.

See the Types section of the reference for more information about the primitive
types:

http://doc.rust-lang.org/reference.html#types
"##

}

register_diagnostics! {
    E0157,
    E0153,
    E0251, // a named type or value has already been imported in this module
    E0252, // a named type or value has already been imported in this module
    E0253, // not directly importable
    E0254, // import conflicts with imported crate in this module
    E0255, // import conflicts with value in this module
    E0256, // import conflicts with type in this module
    E0257, // inherent implementations are only allowed on types defined in the current module
    E0258, // import conflicts with existing submodule
    E0364, // item is private
    E0365  // item is private
}
