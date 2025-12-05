// Regression test for issue #86120.

#![deny(rustdoc::broken_intra_doc_links)]
#![crate_name = "foo"]

pub struct Foo;

/// You should really try [`Self::bar`]!
pub type Bar = Foo;

impl Bar {
    pub fn bar() {}
}

/// The minimum is [`Self::MIN`].
pub type Int = i32;

//@ has foo/type.Bar.html '//a[@href="struct.Foo.html#method.bar"]' 'Self::bar'
//@ has foo/type.Int.html '//a[@href="{{channel}}/std/primitive.i32.html#associatedconstant.MIN"]' 'Self::MIN'
