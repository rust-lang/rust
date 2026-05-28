//@ compile-flags: --document-private-items

#![crate_type = "lib"]

//@ has static/static.FOO.html '//pre' 'static FOO: usize'
static FOO: usize = 1;

//@ has static/static.BAR.html '//pre' 'pub static BAR: usize'
pub static BAR: usize = 1;

//@ has static/static.BAZ.html '//pre' 'pub static mut BAZ: usize'
pub static mut BAZ: usize = 1;
