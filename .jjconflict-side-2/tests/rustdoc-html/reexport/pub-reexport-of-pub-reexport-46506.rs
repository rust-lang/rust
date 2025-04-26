// This is a regression test for <https://github.com/rust-lang/rust/issues/46506>.
// This test ensures that if public re-exported is re-exported, it won't be inlined.

#![crate_name = "foo"]

//@ has 'foo/associations/index.html'
//@ count - '//*[@id="main-content"]/*[@class="section-header"]' 1
//@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Traits'
//@ has - '//*[@id="main-content"]//a[@href="trait.GroupedBy.html"]' 'GroupedBy'
//@ has 'foo/associations/trait.GroupedBy.html'
pub mod associations {
    mod belongs_to {
        pub trait GroupedBy {}
    }
    pub use self::belongs_to::GroupedBy;
}

//@ has 'foo/prelude/index.html'
//@ count - '//*[@id="main-content"]/*[@class="section-header"]' 1
//@ has - '//*[@id="main-content"]/*[@class="section-header"]' 'Re-exports'
//@ has - '//*[@id="main-content"]//*[@id="reexport.GroupedBy"]' 'pub use associations::GroupedBy;'
pub mod prelude {
    pub use associations::GroupedBy;
}
