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
