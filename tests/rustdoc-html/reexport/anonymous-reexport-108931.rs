// Ensuring that anonymous re-exports are always inlined.
// https://github.com/rust-lang/rust/issues/108931

#![crate_name = "foo"]

pub mod foo {
    pub struct Foo;
}

mod bar {
    pub struct Bar;
}

//@ has 'foo/index.html'
// We check that the only "h2" present are "Re-exports" and "Modules".
//@ count - '//*[@id="main-content"]/h2' 2
//@ has - '//*[@id="main-content"]/h2' 'Re-exports'
//@ has - '//*[@id="main-content"]/h2' 'Modules'
//@ has - '//*[@id="main-content"]//*[@class="item-table reexports"]/dt//code' 'pub use foo::Foo as _;'
//@ has - '//*[@id="main-content"]//*[@class="item-table reexports"]/dt//code' 'pub use bar::Bar as _;'
pub use foo::Foo as _;
pub use bar::Bar as _;
