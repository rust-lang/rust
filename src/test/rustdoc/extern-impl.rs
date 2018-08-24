#![crate_name = "foo"]

// @has foo/struct.Foo.html
pub struct Foo;

impl Foo {
    // @has - '//code' 'fn rust0()'
    pub fn rust0() {}
    // @has - '//code' 'fn rust1()'
    pub extern "Rust" fn rust1() {}
    // @has - '//code' 'extern "C" fn c0()'
    pub extern fn c0() {}
    // @has - '//code' 'extern "C" fn c1()'
    pub extern "C" fn c1() {}
    // @has - '//code' 'extern "system" fn system0()'
    pub extern "system" fn system0() {}
}

// @has foo/trait.Bar.html
pub trait Bar {}

// @has - '//code' 'impl Bar for fn()'
impl Bar for fn() {}
// @has - '//code' 'impl Bar for extern "C" fn()'
impl Bar for extern fn() {}
// @has - '//code' 'impl Bar for extern "system" fn()'
impl Bar for extern "system" fn() {}
