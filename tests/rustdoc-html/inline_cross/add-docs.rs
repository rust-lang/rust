//@ aux-build:add-docs.rs

extern crate inner;


//@ has add_docs/struct.MyStruct.html
//@ hasraw add_docs/struct.MyStruct.html "Doc comment from ‘pub use’, Doc comment from definition"
/// Doc comment from 'pub use',
pub use inner::MyStruct;
