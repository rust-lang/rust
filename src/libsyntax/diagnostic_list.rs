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

E0533: r##"
```compile_fail,E0533
#[export_name]
pub fn something() {}
```
"##,

}


register_diagnostics! {
    E0534,
    E0535,
    E0536,
    E0537,
    E0538,
    E0539,
    E0540,
    E0541,
    E0542,
    E0543,
    E0544,
    E0545,
    E0546,
    E0547,
    E0548,
    E0549,
    E0550,
    E0551,
    E0552,
    E0553,
    E0554,
    E0555,
    E0556,
    E0557,
    E0558,
}
