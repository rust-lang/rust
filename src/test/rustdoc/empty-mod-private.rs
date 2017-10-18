// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// compile-flags: --document-private-items

// @has 'empty_mod_private/index.html' '//a[@href="foo/index.html"]' 'foo'
// @has 'empty_mod_private/sidebar-items.js' 'foo'
// @matches 'empty_mod_private/foo/index.html' '//h1' 'Module empty_mod_private::foo'
mod foo {}

// @has 'empty_mod_private/index.html' '//a[@href="bar/index.html"]' 'bar'
// @has 'empty_mod_private/sidebar-items.js' 'bar'
// @matches 'empty_mod_private/bar/index.html' '//h1' 'Module empty_mod_private::bar'
mod bar {
    // @has 'empty_mod_private/bar/index.html' '//a[@href="baz/index.html"]' 'baz'
    // @has 'empty_mod_private/bar/sidebar-items.js' 'baz'
    // @matches 'empty_mod_private/bar/baz/index.html' '//h1' 'Module empty_mod_private::bar::baz'
    mod baz {}
}
