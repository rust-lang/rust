// This test checks two different intra-doc links,
// with the same refdef name, can point at different places.

#![crate_name = "foo"]

//@ has 'foo/index.html'

pub mod one {
    use crate::one as my_mod;
    //@ has 'foo/one/struct.A.html'
    //@ has - '//a[@href="index.html"]' 'there'
    //@ !has - '//a[@href="../two/index.html"]' 'local'
    /// Link [there][my_mod].
    pub struct A;
}

pub mod two {
    use crate::two as my_mod;
    //@ has 'foo/two/struct.A.html'
    //@ has - '//a[@href="../one/index.html"]' 'there'
    //@ has - '//a[@href="index.html"]' 'local'
    /// Link [local][my_mod].
    #[doc(inline)]
    pub use crate::one::A;
}
