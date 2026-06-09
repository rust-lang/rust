// Regression test for <https://github.com/rust-lang/rust/issues/109258>.

#![crate_name = "foo"]

//@ has 'foo/index.html'
// We should only have a "Re-exports" and a "Modules" headers.
//@ count - '//*[@id="main-content"]/h2[@class="section-header"]' 2
//@ has - '//*[@id="main-content"]/h2[@class="section-header"]' 'Re-exports'
//@ has - '//*[@id="main-content"]/h2[@class="section-header"]' 'Modules'

//@ has - '//*[@id="reexport.Foo"]' 'pub use crate::issue_109258::Foo;'
//@ has - '//*[@id="reexport.Foo"]//a[@href="issue_109258/struct.Foo.html"]' 'Foo'
//@ !has 'foo/struct.Foo.html'
pub use crate::issue_109258::Foo;

//@ has 'foo/issue_109258/index.html'
// We should only have a "Structs" header.
//@ count - '//*[@id="main-content"]/h2[@class="section-header"]' 1
//@ has - '//*[@id="main-content"]/h2[@class="section-header"]' 'Structs'
//@ has - '//*[@id="main-content"]//a[@href="struct.Foo.html"]' 'Foo'
//@ has 'foo/issue_109258/struct.Foo.html'
pub mod issue_109258 {
    mod priv_mod {
        pub struct Foo;
    }
    pub use self::priv_mod::Foo;
}
