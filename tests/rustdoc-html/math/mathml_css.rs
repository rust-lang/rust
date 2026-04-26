// This test ensures that the mathml.css file is loaded if,
// and only if, there is a math span in the file.

#![crate_name = "foo"]
#![doc(syntax="+tex_math_dollars")]

//! No math syntax at the crate level,
//! but one of the modules has math syntax in it.

//@ has 'foo/index.html'
//@ matches - '//link/@href' 'mathml'

pub mod no_math_syntax {
    //! No math syntax in this module.
    //@ has 'foo/no_math_syntax/index.html'
    //@ !matches - '//link/@href' 'mathml'

    //@ has 'foo/no_math_syntax/struct.NoMathSyntax.html'
    //@ !matches - '//link/@href' 'mathml'
    /// No math syntax on this struct.
    pub struct NoMathSyntax;
}

pub mod has_math_syntax_direct {
    //! $\sqrt{2}$
    //@ has 'foo/has_math_syntax_direct/index.html'
    //@ matches - '//link/@href' 'mathml'

    //@ has 'foo/has_math_syntax_direct/struct.NoMathSyntax.html'
    //@ !matches - '//link/@href' 'mathml'
    /// No math syntax on this struct.
    pub struct NoMathSyntax;
}

pub mod has_math_syntax_summary {
    //! No math syntax directly on this module.
    //! But there is math syntax on the child struct's first paragraph.

    //@ has 'foo/has_math_syntax_summary/index.html'
    //@ matches - '//link/@href' 'mathml'

    //@ has 'foo/has_math_syntax_summary/struct.MathSyntax.html'
    //@ matches - '//link/@href' 'mathml'
    /// $\sqrt{2}$
    pub struct MathSyntax;
}

pub mod no_math_syntax_summary_because_not_summary {
    //! No math syntax directly on this module.
    //! The child struct has math syntax, but not in its first paragraph.

    //@ has 'foo/no_math_syntax_summary_because_not_summary/index.html'
    //@ !has - '//link/@href' 'mathml'

    //@ has 'foo/no_math_syntax_summary_because_not_summary/struct.MathSyntax.html'
    //@ has - '//link/@href' 'mathml'
    /// Summary paragraph.
    ///
    /// $\sqrt{2}$
    pub struct MathSyntax;
}

pub mod math_syntax_method {
    //! No math syntax directly on this module.
    //! The child struct has math syntax, but not in its first paragraph.

    //@ has 'foo/math_syntax_method/index.html'
    //@ !has - '//link/@href' 'mathml'

    //@ has 'foo/math_syntax_method/struct.MathSyntaxMethod.html'
    //@ has - '//link/@href' 'mathml'
    /// Summary paragraph.
    pub struct MathSyntaxMethod;

    impl MathSyntaxMethod {
        /// $\sqrt{2}$
        pub fn method(self) {}
    }
}
