// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests the rustdoc --sort-modules-by-appearance option, that allows module declarations to appear
// in the order they are declared in the source code, rather than only alphabetically.

// compile-flags: -Z unstable-options --sort-modules-by-appearance

pub mod module_b {}

pub mod module_c {}

pub mod module_a {}

// @matches 'sort_modules_by_appearance/index.html' '(?s)module_b.*module_c.*module_a'
// @matches 'sort_modules_by_appearance/sidebar-items.js' '"module_b".*"module_c".*"module_a"'
