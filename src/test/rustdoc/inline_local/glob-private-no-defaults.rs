// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --no-defaults

#![crate_name = "foo"]

mod mod1 {
    mod mod2 {
        pub struct Mod2Public;
        struct Mod2Private;
    }
    pub use self::mod2::*;

    pub struct Mod1Public;
    struct Mod1Private;
}
pub use mod1::*;

// @has foo/index.html
// @has - "mod1"
// @has - "Mod1Public"
// @!has - "Mod1Private"
// @!has - "mod2"
// @has - "Mod2Public"
// @!has - "Mod2Private"
// @has foo/struct.Mod1Public.html
// @!has foo/struct.Mod1Private.html
// @has foo/struct.Mod2Public.html
// @!has foo/struct.Mod2Private.html

// @has foo/mod1/index.html
// @has - "mod2"
// @has - "Mod1Public"
// @has - "Mod1Private"
// @!has - "Mod2Public"
// @!has - "Mod2Private"
// @has foo/mod1/struct.Mod1Public.html
// @has foo/mod1/struct.Mod1Private.html
// @!has foo/mod1/struct.Mod2Public.html
// @!has foo/mod1/struct.Mod2Private.html

// @has foo/mod1/mod2/index.html
// @has - "Mod2Public"
// @has - "Mod2Private"
// @has foo/mod1/mod2/struct.Mod2Public.html
// @has foo/mod1/mod2/struct.Mod2Private.html

// @!has foo/mod2/index.html
// @!has foo/mod2/struct.Mod2Public.html
// @!has foo/mod2/struct.Mod2Private.html
