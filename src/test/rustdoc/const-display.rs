#![crate_name = "foo"]

#![unstable(feature = "humans",
            reason = "who ever let humans program computers, we're apparently really bad at it",
            issue = "0")]

#![feature(rustc_const_unstable, const_fn, foo, foo2)]
#![feature(staged_api)]

// @has 'foo/fn.foo.html' '//pre' 'pub unsafe fn foo() -> u32'
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature="foo")]
pub const unsafe fn foo() -> u32 { 42 }

// @has 'foo/fn.foo2.html' '//pre' 'pub fn foo2() -> u32'
#[unstable(feature = "humans", issue="0")]
pub const fn foo2() -> u32 { 42 }

// @has 'foo/fn.bar2.html' '//pre' 'pub const fn bar2() -> u32'
#[stable(feature = "rust1", since = "1.0.0")]
pub const fn bar2() -> u32 { 42 }

// @has 'foo/fn.foo2_gated.html' '//pre' 'pub unsafe fn foo2_gated() -> u32'
#[unstable(feature = "foo2", issue="0")]
pub const unsafe fn foo2_gated() -> u32 { 42 }

// @has 'foo/fn.bar2_gated.html' '//pre' 'pub const unsafe fn bar2_gated() -> u32'
#[stable(feature = "rust1", since = "1.0.0")]
pub const unsafe fn bar2_gated() -> u32 { 42 }

// @has 'foo/fn.bar_not_gated.html' '//pre' 'pub unsafe fn bar_not_gated() -> u32'
pub const unsafe fn bar_not_gated() -> u32 { 42 }
