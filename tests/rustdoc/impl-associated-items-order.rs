// This test ensures that impl associated items always follow this order:
//
// 1. Consts
// 2. Types
// 3. Functions

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]
#![crate_name = "foo"]

//@ has 'foo/struct.Bar.html'
pub struct Bar;

impl Bar {
    //@ has - '//*[@id="implementations-list"]//*[@class="impl-items"]/section[3]/h4' \
    // 'pub fn foo()'
    pub fn foo() {}
    //@ has - '//*[@id="implementations-list"]//*[@class="impl-items"]/section[1]/h4' \
    // 'pub const X: u8 = 12u8'
    pub const X: u8 = 12;
    //@ has - '//*[@id="implementations-list"]//*[@class="impl-items"]/section[2]/h4' \
    // 'pub type Y = u8'
    pub type Y = u8;
}

pub trait Foo {
    const W: u32;
    fn yeay();
    type Z;
}

impl Foo for Bar {
    //@ has - '//*[@id="trait-implementations-list"]//*[@class="impl-items"]/section[2]/h4' \
    // 'type Z = u8'
    type Z = u8;
    //@ has - '//*[@id="trait-implementations-list"]//*[@class="impl-items"]/section[1]/h4' \
    // 'const W: u32 = 12u32'
    const W: u32 = 12;
    //@ has - '//*[@id="trait-implementations-list"]//*[@class="impl-items"]/section[3]/h4' \
    // 'fn yeay()'
    fn yeay() {}
}
