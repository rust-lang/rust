// aux-build:empty.rs
// aux-build:variant-struct.rs
// build-aux-docs
// ignore-cross-compile

// @has issue_33178/index.html
// @has - //a/@title empty
// @has - //a/@href ../empty/index.html
pub extern crate empty;

// @has - //a/@title variant_struct
// @has - //a/@href ../variant_struct/index.html
pub extern crate variant_struct as foo;
