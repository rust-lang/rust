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

register_diagnostics! {
    E0157,
    E0153,
    E0251, // a named type or value has already been imported in this module
    E0252, // a named type or value has already been imported in this module
    E0253, // not directly importable
    E0254, // import conflicts with imported crate in this module
    E0255, // import conflicts with value in this module
    E0256, // import conflicts with type in this module
    E0257, // inherent implementations are only allowen on types defined in the current module
    E0258, // import conflicts with existing submodule
    E0259, // an extern crate has already been imported into this module
    E0260 // name conflicts with an external crate that has been imported into this module
}

__build_diagnostic_array! { DIAGNOSTICS }
