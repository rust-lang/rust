//@ aux-build:trait-vis.rs

extern crate inner;

//@ has trait_vis/type.SomeStruct.html
//@ has - '//h3[@class="code-header"]' 'impl Clone for SomeStruct'
pub use inner::SomeStruct;
