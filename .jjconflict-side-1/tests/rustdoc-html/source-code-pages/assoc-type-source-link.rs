// This test ensures that the source links are generated for impl associated types.

#![crate_name = "foo"]
#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

//@ has 'foo/struct.Bar.html'
pub struct Bar;

impl Bar {
    //@ has - '//*[@id="implementations-list"]//*[@id="associatedtype.Y"]/a' 'Source'
    //@ has - '//*[@id="implementations-list"]//*[@id="associatedtype.Y"]/a/@href' \
    // '../src/foo/assoc-type-source-link.rs.html#14'
    pub type Y = u8;
}

pub trait Foo {
    type Z;
}

impl Foo for Bar {
    //@ has - '//*[@id="trait-implementations-list"]//*[@id="associatedtype.Z"]/a' 'Source'
    //@ has - '//*[@id="trait-implementations-list"]//*[@id="associatedtype.Z"]/a/@href' \
    // '../src/foo/assoc-type-source-link.rs.html#25'
    type Z = u8;
}
