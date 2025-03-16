#![allow(unused)]
#![warn(clippy::suspicious_doc_comments)]
#![allow(clippy::empty_line_after_doc_comments)]

//! Real module documentation.
///! Fake module documentation.
//~^ suspicious_doc_comments
fn baz() {}

pub mod singleline_outer_doc {
    ///! This module contains useful functions.
    //~^ suspicious_doc_comments

    pub fn bar() {}
}

pub mod singleline_inner_doc {
    //! This module contains useful functions.

    pub fn bar() {}
}

pub mod multiline_outer_doc {
    /**! This module contains useful functions.
     */
    //~^^ suspicious_doc_comments

    pub fn bar() {}
}

pub mod multiline_inner_doc {
    /*! This module contains useful functions.
     */

    pub fn bar() {}
}

pub mod multiline_outer_doc2 {
    ///! This module
    //~^ suspicious_doc_comments
    ///! contains
    ///! useful functions.

    pub fn bar() {}
}

pub mod multiline_outer_doc3 {
    ///! a
    //~^ suspicious_doc_comments
    ///! b

    /// c
    pub fn bar() {}
}

pub mod multiline_outer_doc4 {
    ///! a
    //~^ suspicious_doc_comments
    /// b
    pub fn bar() {}
}

pub mod multiline_outer_doc_gap {
    ///! a
    //~^ suspicious_doc_comments

    ///! b
    pub fn bar() {}
}

pub mod multiline_outer_doc_commented {
    /////! This outer doc comment was commented out.
    pub fn bar() {}
}

pub mod outer_doc_macro {
    ///! Very cool macro
    //~^ suspicious_doc_comments
    macro_rules! x {
        () => {};
    }
}

pub mod useless_outer_doc {
    ///! Huh.
    //~^ suspicious_doc_comments
    use std::mem;
}

fn main() {}
