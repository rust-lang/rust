// This test checks that inlining a `no_inline` is done correctly.

#![crate_name = "foo"]

//@ has 'foo/index.html'
// There should be `Re-exports` and `Structs`
//@ count - '//*[@id="main-content"]/h2' 2
//@ has - '//*[@id="main-content"]/h2[@class="section-header"]' 'Re-exports'
//@ has - '//*[@id="main-content"]/h2[@class="section-header"]' 'Structs'

//@ has - '//*[@id="main-content"]/dl[@class="item-table reexports"]/dt' 'pub use self::bar::A;'
//@ has - '//*[@id="main-content"]/dl[@class="item-table"]/dt' 'X'

mod bar {
    pub struct A;
}

#[doc(no_inline)]
pub use self::bar::A;
#[doc(inline)]
pub use self::A as X;
