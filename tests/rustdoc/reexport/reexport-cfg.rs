// This test ensures that only the re-export `cfg` will be displayed and that it won't
// include `cfg`s from the previous chained items.

#![crate_name = "foo"]
#![feature(doc_cfg)]

mod foo {
    #[cfg(not(feature = "foo"))]
    pub struct Bar;

    #[doc(cfg(not(feature = "bar")))]
    pub struct Bar2;
}

//@ has 'foo/index.html'
//@ has - '//dt' 'BabarNon-lie'
#[cfg(not(feature = "lie"))]
pub use crate::foo::Bar as Babar;

//@ has - '//dt' 'Babar2Non-cake'
#[doc(cfg(not(feature = "cake")))]
pub use crate::foo::Bar2 as Babar2;

//@ has - '//*[@class="item-table reexports"]/dt' 'pub use crate::Babar as Elephant;Non-robot'
#[cfg(not(feature = "robot"))]
pub use crate::Babar as Elephant;

//@ has - '//*[@class="item-table reexports"]/dt' 'pub use crate::Babar2 as Elephant2;Non-cat'
#[doc(cfg(not(feature = "cat")))]
pub use crate::Babar2 as Elephant2;
