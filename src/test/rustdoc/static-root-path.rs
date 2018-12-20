// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-Z unstable-options --static-root-path /cache/

// @has static_root_path/struct.SomeStruct.html
// @matches - '"/cache/main\.js"'
// @!matches - '"\.\./main\.js"'
// @matches - '"\.\./search-index\.js"'
// @!matches - '"/cache/search-index\.js"'
pub struct SomeStruct;

// @has src/static_root_path/static-root-path.rs.html
// @matches - '"/cache/source-script\.js"'
// @!matches - '"\.\./\.\./source-script\.js"'
// @matches - '"\.\./\.\./source-files.js"'
// @!matches - '"/cache/source-files\.js"'
