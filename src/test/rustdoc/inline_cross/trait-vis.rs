// aux-build:trait-vis.rs

extern crate inner;

// @has trait_vis/struct.SomeStruct.html
// @has - '//code' 'impl Clone for SomeStruct'
pub use inner::SomeStruct;
