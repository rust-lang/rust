// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! doc {
    (#[$outer:meta] mod $i:ident { #![$inner:meta] }) =>
    (
        #[$outer]
        pub mod $i {
            #![$inner]
        }
    )
}

doc! {
    /// Outer comment
    mod Foo {
        //! Inner comment
    }
}

// @has issue_23812/Foo/index.html
// @has - 'Outer comment'
// @!has - '/// Outer comment'
// @has - 'Inner comment'
// @!has - '//! Inner comment'


doc! {
    /** Outer block comment */
    mod Bar {
        /*! Inner block comment */
    }
}

// @has issue_23812/Bar/index.html
// @has - 'Outer block comment'
// @!has - '/** Outer block comment */'
// @has - 'Inner block comment'
// @!has - '/*! Inner block comment */'
