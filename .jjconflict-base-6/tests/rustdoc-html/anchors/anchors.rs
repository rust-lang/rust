// This test ensures that anchors are generated in the right places.

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]
#![crate_name = "foo"]

pub struct Foo;

//@ has 'foo/trait.Bar.html'
pub trait Bar {
    // There should be no anchors here.
    //@ snapshot no_type_anchor - '//*[@id="associatedtype.T"]'
    type T;
    // There should be no anchors here.
    //@ snapshot no_const_anchor - '//*[@id="associatedconstant.YOLO"]'
    const YOLO: u32;

    // There should be no anchors here.
    //@ snapshot no_tymethod_anchor - '//*[@id="tymethod.foo"]'
    fn foo();
    // There should be no anchors here.
    //@ snapshot no_trait_method_anchor - '//*[@id="method.bar"]'
    fn bar() {}
}

//@ has 'foo/struct.Foo.html'
impl Bar for Foo {
    //@ has - '//*[@id="associatedtype.T"]/a[@class="anchor"]' ''
    type T = u32;
    //@ has - '//*[@id="associatedconstant.YOLO"]/a[@class="anchor"]' ''
    const YOLO: u32 = 0;

    //@ has - '//*[@id="method.foo"]/a[@class="anchor"]' ''
    fn foo() {}
    // Same check for provided "bar" method.
    //@ has - '//*[@id="method.bar"]/a[@class="anchor"]' ''
}

impl Foo {
    //@ snapshot no_const_anchor2 - '//*[@id="associatedconstant.X"]'
    // There should be no anchors here.
    pub const X: i32 = 0;
    //@ snapshot no_type_anchor2 - '//*[@id="associatedtype.Y"]'
    // There should be no anchors here.
    pub type Y = u32;
    //@ snapshot no_method_anchor - '//*[@id="method.new"]'
    // There should be no anchors here.
    pub fn new() -> Self { Self }
}
