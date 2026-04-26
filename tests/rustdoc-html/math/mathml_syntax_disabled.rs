// This test ensures that the mathml.css file is loaded if,
// and only if, there is a math span in the file.

#![crate_name = "foo"]
#![doc(syntax="-tex_math_dollars")]

//! $\sqrt{2}$
//@ has 'foo/index.html'
//@ !matches - '//link/@href' 'mathml'

pub mod module {
    //! $\sqrt{2}$
    //@ has 'foo/module/index.html'
    //@ !matches - '//link/@href' 'mathml'

    //@ has 'foo/module/struct.MathSyntax.html'
    //@ !matches - '//link/@href' 'mathml'
    /// $\sqrt{2}$
    pub struct MathSyntax;

    impl MathSyntax {
        /// $\sqrt{2}$
        pub fn method(self) {}
    }
}
