#![crate_name = "foo"]

#![unstable(feature = "humans",
            reason = "who ever let humans program computers, we're apparently really bad at it",
            issue = "none")]

#![feature(foo, foo2)]
#![feature(staged_api)]

// @has 'foo/fn.foo.html' '//pre' 'pub unsafe fn foo() -> u32'
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature="foo", issue = "none")]
pub const unsafe fn foo() -> u32 { 42 }

// @has 'foo/fn.foo2.html' '//pre' 'pub const fn foo2() -> u32'
#[unstable(feature = "humans", issue = "none")]
pub const fn foo2() -> u32 { 42 }

// @has 'foo/fn.bar2.html' '//pre' 'pub const fn bar2() -> u32'
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
pub const fn bar2() -> u32 { 42 }

// @has 'foo/fn.foo2_gated.html' '//pre' 'pub const unsafe fn foo2_gated() -> u32'
#[unstable(feature = "foo2", issue = "none")]
pub const unsafe fn foo2_gated() -> u32 { 42 }

// @has 'foo/fn.bar2_gated.html' '//pre' 'pub const unsafe fn bar2_gated() -> u32'
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
pub const unsafe fn bar2_gated() -> u32 { 42 }

// @has 'foo/fn.bar_not_gated.html' '//pre' 'pub const unsafe fn bar_not_gated() -> u32'
pub const unsafe fn bar_not_gated() -> u32 { 42 }

pub struct Foo;

impl Foo {
    // @has 'foo/struct.Foo.html' '//h4[@id="method.gated"]/code' 'pub unsafe fn gated() -> u32'
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature="foo", issue = "none")]
    pub const unsafe fn gated() -> u32 { 42 }
}
