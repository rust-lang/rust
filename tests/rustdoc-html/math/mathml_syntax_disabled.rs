// This test ensures that the mathml.css file is loaded if,
// and only if, there is a math span in the file.

#![feature(rustdoc_texmath)]
#![crate_name = "foo"]
#![doc(syntax="-tex_math_dollars")]

//! $\sqrt{2}$
//@ has 'foo/index.html'
//@ !matches - '//link/@href' 'mathml'
//@ count - '//math' 0

pub mod module {
    //! $\sqrt{2}$
    //@ has 'foo/module/index.html'
    //@ !matches - '//link/@href' 'mathml'
    //@ count - '//math' 0

    //@ has 'foo/module/struct.MathSyntax.html'
    //@ !matches - '//link/@href' 'mathml'
    //@ count - '//math' 0
    /// $\sqrt{2}$
    pub struct MathSyntax;

    impl MathSyntax {
        /// $\sqrt{2}$
        pub fn method(self) {}
    }
}
