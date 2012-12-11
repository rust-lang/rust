// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pp-exact

// some single-line non-doc comment

/// some single line outer-docs
fn a() { }

fn b() {
    //! some single line inner-docs
}

/*
 * some multi-line non-doc comment
 */

/**
 * some multi-line outer-docs
 */
fn c() { }

fn d() {
    /*!
     * some multi-line inner-docs
     */
}

#[doc = "unsugared outer doc-comments work also"]
fn e() { }

fn f() {
    #[doc = "as do inner ones"];
}
