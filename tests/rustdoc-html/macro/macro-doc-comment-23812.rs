// https://github.com/rust-lang/rust/issues/23812
#![crate_name="issue_23812"]

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

//@ has issue_23812/Foo/index.html
//@ hasraw - 'Outer comment'
//@ !hasraw - '/// Outer comment'
//@ hasraw - 'Inner comment'
//@ !hasraw - '//! Inner comment'


doc! {
    /** Outer block comment */
    mod Bar {
        /*! Inner block comment */
    }
}

//@ has issue_23812/Bar/index.html
//@ hasraw - 'Outer block comment'
//@ !hasraw - '/** Outer block comment */'
//@ hasraw - 'Inner block comment'
//@ !hasraw - '/*! Inner block comment */'
