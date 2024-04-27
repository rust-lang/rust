#![crate_type = "lib"]
#![feature(staged_api)]
#![stable(feature = "foo", since = "1.0.0")]

#[stable(feature = "foo", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_foo", issue = "none")]
pub fn foo() {}
//~^ ERROR attributes `#[rustc_const_unstable]` and `#[rustc_const_stable]` require the function or method to be `const`

#[stable(feature = "bar", since = "1.0.0")]
#[rustc_const_stable(feature = "const_bar", since = "1.0.0")]
pub fn bar() {}
//~^ ERROR attributes `#[rustc_const_unstable]` and `#[rustc_const_stable]` require the function or method to be `const`

#[stable(feature = "potato", since = "1.0.0")]
pub struct Potato;

impl Potato {
    #[stable(feature = "salad", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_salad", issue = "none")]
    pub fn salad(&self) -> &'static str { "mmmmmm" }
    //~^ ERROR attributes `#[rustc_const_unstable]` and `#[rustc_const_stable]` require the function or method to be `const`

    #[stable(feature = "roasted", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_roasted", issue = "none")]
    pub fn roasted(&self) -> &'static str { "mmmmmmmmmm" }
    //~^ ERROR attributes `#[rustc_const_unstable]` and `#[rustc_const_stable]` require the function or method to be `const`
}

#[stable(feature = "bar", since = "1.0.0")]
#[rustc_const_stable(feature = "const_bar", since = "1.0.0")]
pub extern "C" fn bar_c() {}
//~^ ERROR attributes `#[rustc_const_unstable]` and `#[rustc_const_stable]` require the function or method to be `const`

#[stable(feature = "foo", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_foo", issue = "none")]
pub extern "C" fn foo_c() {}
//~^ ERROR attributes `#[rustc_const_unstable]` and `#[rustc_const_stable]` require the function or method to be `const`


#[stable(feature = "foobar", since = "1.0.0")]
#[rustc_const_unstable(feature = "foobar_const", issue = "none")]
pub const fn foobar() {}

#[stable(feature = "barfoo", since = "1.0.0")]
#[rustc_const_stable(feature = "barfoo_const", since = "1.0.0")]
pub const fn barfoo() {}
