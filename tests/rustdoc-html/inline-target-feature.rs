//@ only-x86_64
//@ aux-build:inline-target-feature.rs
//@ build-aux-docs

#![feature(doc_cfg)]
#![crate_name = "foo"]

extern crate inline_target_feature;

//@ has foo/fn.foo.html
//@ has - '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' 'Available with target feature avx only.'
pub use inline_target_feature::foo;
