//@ aux-build: primitive-reexport.rs
//@ compile-flags: --extern foo
//@ edition: 2018

#![crate_name = "bar"]

//@ has bar/p/index.html
//@ has - '//code' 'pub use bool;'
//@ has - '//code/a[@href="{{channel}}/std/primitive.bool.html"]' 'bool'
//@ has - '//code' 'pub use char as my_char;'
//@ has - '//code/a[@href="{{channel}}/std/primitive.char.html"]' 'char'
pub mod p {
    pub use foo::bar::*;
}

//@ has bar/baz/index.html
//@ has - '//code' 'pub use bool;'
//@ has - '//code/a[@href="{{channel}}/std/primitive.bool.html"]' 'bool'
//@ has - '//code' 'pub use char as my_char;'
//@ has - '//code/a[@href="{{channel}}/std/primitive.char.html"]' 'char'
pub use foo::bar as baz;

//@ has bar/index.html
//@ has - '//code' 'pub use str;'
//@ has - '//code/a[@href="{{channel}}/std/primitive.str.html"]' 'str'
//@ has - '//code' 'pub use i32 as my_i32;'
//@ has - '//code/a[@href="{{channel}}/std/primitive.i32.html"]' 'i32'
pub use str;
pub use i32 as my_i32;
