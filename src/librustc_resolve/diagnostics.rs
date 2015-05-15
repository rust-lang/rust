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

E0251: r##"
Two items of the same name cannot be imported without rebinding one of the
items under a new local name.

An example of this error:

```
use foo::baz;
use bar::*; // error, do `use foo::baz as quux` instead on the previous line

fn main() {}

mod foo {
    pub struct baz;
}

mod bar {
    pub mod baz {}
}
```
"##,

E0252: r##"
Two items of the same name cannot be imported without rebinding one of the
items under a new local name.

An example of this error:

```
use foo::baz;
use bar::baz; // error, do `use bar::baz as quux` instead

fn main() {}

mod foo {
    pub struct baz;
}

mod bar {
    pub mod baz {}
}
```
"##,

E0255: r##"
You can't import a value whose name is the same as another value defined in the
module.

An example of this error:

```
use bar::foo; // error, do `use bar::foo as baz` instead

fn foo() {}

mod bar {
     pub fn foo() {}
}

fn main() {}
```
"##,

E0256: r##"
You can't import a type or module when the name of the item being imported is
the same as another type or submodule defined in the module.

An example of this error:

```
use foo::Bar; // error

type Bar = u32;

mod foo {
    pub mod Bar { }
}

fn main() {}
```
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
    E0253, // not directly importable
    E0254, // import conflicts with imported crate in this module
    E0257,
    E0258,
    E0364, // item is private
    E0365  // item is private
}
