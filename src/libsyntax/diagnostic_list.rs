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

fn main() {}
```
"##,

}

register_diagnostics! {
    E0534, // expected one argument
    E0535, // invalid argument
    E0536, // expected 1 cfg-pattern
    E0537, // invalid predicate
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
