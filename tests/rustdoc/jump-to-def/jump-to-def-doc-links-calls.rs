//@ compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

//@ has 'src/foo/jump-to-def-doc-links-calls.rs.html'

//@ has - '//a[@href="../../foo/struct.Bar.html"]' 'Bar'
pub struct Bar;

impl std::default::Default for Bar {
    //@ has - '//a[@href="#20-22"]' 'new'
    fn default() -> Self {
        Self::new()
    }
}

//@ has - '//a[@href="#8"]' 'Bar'
impl Bar {
     //@ has - '//a[@href="#24-26"]' 'bar'
     pub fn new()-> Self {
         Self::bar()
     }

     pub fn bar() -> Self {
         Self
     }
}
