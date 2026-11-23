//@ aux-build: reexport-glob-cross-crate.rs

#![crate_name = "foo"]

extern crate reexport_glob_cross_crate;

//@ has 'foo/index.html'
// We check that the only "h2" present is "Re-exports".
//@ count - '//*[@id="main-content"]/h2' 1
//@ has - '//*[@id="main-content"]/h2' 'Re-exports'
//@ has - '//*[@id="main-content"]//*[@class="item-table reexports"]/dt//code' \
//        'pub use reexport_glob_cross_crate::prelude::TraitA as _;'
//@ has - '//*[@id="main-content"]//*[@class="item-table reexports"]/dt//code' \
//        'pub use reexport_glob_cross_crate::prelude::TraitB as _;'
pub use reexport_glob_cross_crate::prelude::*;
