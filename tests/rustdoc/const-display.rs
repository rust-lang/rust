#![crate_name = "foo"]

#![stable(feature = "rust1", since = "1.0.0")]

#![feature(foo, foo2)]
#![feature(staged_api)]

//@ has 'foo/fn.foo.html' '//pre' 'pub fn foo() -> u32'
//@ has - '//span[@class="since"]' '1.0.0 (const: unstable)'
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature="foo", issue = "none")]
pub const fn foo() -> u32 { 42 }

//@ has 'foo/fn.foo_unsafe.html' '//pre' 'pub unsafe fn foo_unsafe() -> u32'
//@ has - '//span[@class="since"]' '1.0.0 (const: unstable)'
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature="foo", issue = "none")]
pub const unsafe fn foo_unsafe() -> u32 { 42 }

//@ has 'foo/fn.foo2.html' '//pre' 'pub const fn foo2() -> u32'
//@ !hasraw - '//span[@class="since"]'
#[unstable(feature = "humans", issue = "none")]
pub const fn foo2() -> u32 { 42 }

//@ has 'foo/fn.foo3.html' '//pre' 'pub const fn foo3() -> u32'
//@ !hasraw - '//span[@class="since"]'
#[unstable(feature = "humans", issue = "none")]
#[rustc_const_unstable(feature = "humans", issue = "none")]
pub const fn foo3() -> u32 { 42 }

//@ has 'foo/fn.bar2.html' '//pre' 'pub const fn bar2() -> u32'
//@ has - //span '1.0.0 (const: 1.0.0)'
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
pub const fn bar2() -> u32 { 42 }


//@ has 'foo/fn.foo2_gated.html' '//pre' 'pub const unsafe fn foo2_gated() -> u32'
//@ !hasraw - '//span[@class="since"]'
#[unstable(feature = "foo2", issue = "none")]
pub const unsafe fn foo2_gated() -> u32 { 42 }

//@ has 'foo/fn.bar2_gated.html' '//pre' 'pub const unsafe fn bar2_gated() -> u32'
//@ has - '//span[@class="since"]' '1.0.0 (const: 1.0.0)'
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
pub const unsafe fn bar2_gated() -> u32 { 42 }

#[unstable(
    feature = "humans",
    reason = "who ever let humans program computers, we're apparently really bad at it",
    issue = "none",
)]
pub mod unstable {
    //@ has 'foo/unstable/fn.bar_not_gated.html' '//pre' 'pub const unsafe fn bar_not_gated() -> u32'
    //@ !hasraw - '//span[@class="since"]'
    pub const unsafe fn bar_not_gated() -> u32 { 42 }
}

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Foo;

impl Foo {
    //@ has 'foo/struct.Foo.html' '//*[@id="method.gated"]/h4[@class="code-header"]' 'pub fn gated() -> u32'
    //@ has - '//span[@class="since"]' '1.0.0 (const: unstable)'
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature="foo", issue = "none")]
    pub const fn gated() -> u32 { 42 }

    //@ has 'foo/struct.Foo.html' '//*[@id="method.gated_unsafe"]/h4[@class="code-header"]' 'pub unsafe fn gated_unsafe() -> u32'
    //@ has - '//span[@class="since"]' '1.0.0 (const: unstable)'
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature="foo", issue = "none")]
    pub const unsafe fn gated_unsafe() -> u32 { 42 }

    //@ has 'foo/struct.Foo.html' '//*[@id="method.stable_impl"]/h4[@class="code-header"]' 'pub const fn stable_impl() -> u32'
    //@ has - '//span[@class="since"]' '1.0.0 (const: 1.2.0)'
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const2", since = "1.2.0")]
    pub const fn stable_impl() -> u32 { 42 }
}

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Bar;

impl Bar {
    // Show non-const stabilities that are the same as the enclosing item.
    //@ has 'foo/struct.Bar.html' '//span[@class="since"]' '1.0.0 (const: 1.2.0)'
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const2", since = "1.2.0")]
    pub const fn stable_impl() -> u32 { 42 }
}
