#![crate_name = "foo"]
#![feature(rustdoc_internals)]

// tests for the html <title> element

// @has foo/index.html '//head/title' 'foo - Rust'

// @has foo/fn.widget_count.html '//head/title' 'widget_count in foo - Rust'
/// blah
pub fn widget_count() {}

// @has foo/struct.Widget.html '//head/title' 'Widget in foo - Rust'
pub struct Widget;

// @has foo/constant.ANSWER.html '//head/title' 'ANSWER in foo - Rust'
pub const ANSWER: u8 = 42;

// @has foo/blah/index.html '//head/title' 'foo::blah - Rust'
pub mod blah {
    // @has foo/blah/struct.Widget.html '//head/title' 'Widget in foo::blah - Rust'
    pub struct Widget;

    // @has foo/blah/trait.Awesome.html '//head/title' 'Awesome in foo::blah - Rust'
    pub trait Awesome {}

    // @has foo/blah/fn.make_widget.html '//head/title' 'make_widget in foo::blah - Rust'
    pub fn make_widget() {}

    // @has foo/macro.cool_macro.html '//head/title' 'cool_macro in foo - Rust'
    #[macro_export]
    macro_rules! cool_macro {
        ($t:tt) => { $t }
    }
}

// @has foo/keyword.continue.html '//head/title' 'continue - Rust'
#[doc(keyword = "continue")]
mod continue_keyword {}

// @has foo/primitive.u8.html '//head/title' 'u8 - Rust'
// @!has - '//head/title' 'foo'
#[doc(primitive = "u8")]
/// `u8` docs
mod u8 {}
