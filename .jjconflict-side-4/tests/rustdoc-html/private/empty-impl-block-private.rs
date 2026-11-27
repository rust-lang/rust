#![feature(inherent_associated_types)]
#![allow(incomplete_features)]
#![crate_name = "foo"]

//@ has 'foo/struct.Foo.html'
pub struct Foo;

// There are 3 impl blocks with public item and one that should not be displayed
// because it only contains private items.
//@ count - '//*[@class="impl"]' 'impl Foo' 3

// Impl block only containing private items should not be displayed.
/// Private
impl Foo {
    const BAR: u32 = 0;
    type FOO = i32;
    fn hello() {}
}

// But if any element of the impl block is public, it should be displayed.
/// Not private
impl Foo {
    pub const BAR: u32 = 0;
    type FOO = i32;
    fn hello() {}
}

/// Not private
impl Foo {
    const BAR: u32 = 0;
    pub type FOO = i32;
    fn hello() {}
}

/// Not private
impl Foo {
    const BAR: u32 = 0;
    type FOO = i32;
    pub fn hello() {}
}
