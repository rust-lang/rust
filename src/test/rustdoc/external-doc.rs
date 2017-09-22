// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(external_doc)]

// @has external_doc/struct.CanHasDocs.html
// @has - '//h1' 'External Docs'
// @has - '//h2' 'Inline Docs'
#[doc(include = "auxiliary/external-doc.md")]
/// ## Inline Docs
pub struct CanHasDocs;
