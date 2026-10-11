//@ aux-build:add-docs.rs

extern crate inner;


//@ has add_docs/type.MyStruct.html
//@ hasraw add_docs/type.MyStruct.html "Doc comment from ‘pub use’, Doc comment from definition"
/// Doc comment from 'pub use',
pub use inner::MyStruct;
