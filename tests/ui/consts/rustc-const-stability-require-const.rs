#![crate_type = "lib"]
#![feature(staged_api, rustc_attrs)]
#![stable(feature = "foo", since = "1.0.0")]

#[stable(feature = "foo", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_foo", issue = "none")]
pub fn foo() {}
//~^ ERROR require the function or method to be `const`

#[stable(feature = "bar", since = "1.0.0")]
#[rustc_const_stable(feature = "const_bar", since = "1.0.0")]
pub fn bar() {}
//~^ ERROR require the function or method to be `const`

#[stable(feature = "potato", since = "1.0.0")]
pub struct Potato;

impl Potato {
    #[stable(feature = "salad", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_salad", issue = "none")]
    pub fn salad(&self) -> &'static str { "mmmmmm" }
    //~^ ERROR require the function or method to be `const`

    #[stable(feature = "roasted", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_roasted", issue = "none")]
    pub fn roasted(&self) -> &'static str { "mmmmmmmmmm" }
    //~^ ERROR require the function or method to be `const`
}

#[stable(feature = "bar", since = "1.0.0")]
#[rustc_const_stable(feature = "const_bar", since = "1.0.0")]
pub extern "C" fn bar_c() {}
//~^ ERROR require the function or method to be `const`

#[stable(feature = "foo", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_foo", issue = "none")]
pub extern "C" fn foo_c() {}
//~^ ERROR require the function or method to be `const`


#[stable(feature = "foobar", since = "1.0.0")]
#[rustc_const_unstable(feature = "foobar_const", issue = "none")]
pub const fn foobar() {}

#[stable(feature = "barfoo", since = "1.0.0")]
#[rustc_const_stable(feature = "barfoo_const", since = "1.0.0")]
pub const fn barfoo() {}

// `rustc_const_stable` also requires the function to be stable.

#[rustc_const_stable(feature = "barfoo_const", since = "1.0.0")]
const fn barfoo_unmarked() {}
//~^ ERROR can only be applied to functions that are declared `#[stable]`

#[unstable(feature = "unstable", issue = "none")]
#[rustc_const_stable(feature = "barfoo_const", since = "1.0.0")]
pub const fn barfoo_unstable() {}
//~^ ERROR can only be applied to functions that are declared `#[stable]`
