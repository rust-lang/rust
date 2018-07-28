// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Rustdoc would previously report resolution failures on items that weren't in the public docs.
// These failures were legitimate, but not truly relevant - the docs in question couldn't be
// checked for accuracy anyway.

#![deny(intra_doc_link_resolution_failure)]

/// ooh, i'm a [rebel] just for kicks
struct SomeStruct;
