#![crate_name = "foo"]

//@ has 'foo/struct.Foo.html'
//@ has - '//*[@id="deref-methods-i32"]' 'Methods from Deref<Target = i32>'
//@ has - '//*[@id="deref-methods-i32-1"]//*[@id="associatedconstant.BITS"]/h4' \
//        'pub const BITS: u32 = 32u32'
pub struct Foo(i32);

impl std::ops::Deref for Foo {
    type Target = i32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
