#![feature(staged_api)]
#![stable(feature = "rust1", since = "1.0.0")]
#![crate_type = "lib"]

#[stable(feature = "rust1", since = "1.0.0")]
mod mod_stable {}
//~^ ERROR module is private but has a stability attribute

#[unstable(feature = "foo", issue = "none")]
mod mod_unstable {}
//~^ ERROR module is private but has a stability attribute

#[stable(feature = "rust1", since = "1.0.0")]
fn fn_stable() {}
//~^ ERROR function is private but has a stability attribute

#[unstable(feature = "foo", issue = "none")]
fn fn_unstable() {}
//~^ ERROR function is private but has a stability attribute

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
const fn const_fn_stable() {}
//~^ ERROR function is private but has a stability attribute

#[unstable(feature = "foo", issue = "none")]
#[rustc_const_unstable(feature = "foo", issue = "none")]
const fn const_fn_unstable() {}
//~^ ERROR function is private but has a stability attribute
