// Ensuring that anonymous re-exports are never inlined but still displayed.
// https://github.com/rust-lang/rust/issues/108931

#![crate_name = "foo"]

//@ has 'foo/index.html'
// We check that the only "h2" present is "Re-exports".
//@ count - '//*[@id="main-content"]/h2' 1
//@ has - '//*[@id="main-content"]/h2' 'Re-exports'
//@ has - '//*[@id="main-content"]//*[@class="item-table reexports"]/dt//code' \
//        'pub use crate::ext::Foo as _;'
//@ has - '//*[@id="main-content"]//*[@class="item-table reexports"]/dt//code' \
//        'pub use crate::ext::Bar as _;'

mod ext {
    pub trait Foo {}
    pub trait Bar {}
}

mod b {
    pub use crate::ext::Foo as _;
    pub use crate::ext::Bar as _;
}

pub use crate::b::*;
