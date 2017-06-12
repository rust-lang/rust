// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// @has 'empty_mod_public/index.html' '//a[@href="foo/index.html"]' 'foo'
// @has 'empty_mod_public/sidebar-items.js' 'foo'
// @matches 'empty_mod_public/foo/index.html' '//h1' 'Module empty_mod_public::foo'
pub mod foo {}

// @has 'empty_mod_public/index.html' '//a[@href="bar/index.html"]' 'bar'
// @has 'empty_mod_public/sidebar-items.js' 'bar'
// @matches 'empty_mod_public/bar/index.html' '//h1' 'Module empty_mod_public::bar'
pub mod bar {
    // @has 'empty_mod_public/bar/index.html' '//a[@href="baz/index.html"]' 'baz'
    // @has 'empty_mod_public/bar/sidebar-items.js' 'baz'
    // @matches 'empty_mod_public/bar/baz/index.html' '//h1' 'Module empty_mod_public::bar::baz'
    pub mod baz {}
}
