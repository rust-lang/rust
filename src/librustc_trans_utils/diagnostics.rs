// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
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

E0558: r##"
The `export_name` attribute was malformed.

Erroneous code example:

```ignore (error-emitted-at-codegen-which-cannot-be-handled-by-compile_fail)
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

}
