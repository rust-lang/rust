#![crate_type = "lib"]
#![feature(staged_api)]
#![stable(feature = "stability_attribute_implies", since = "1.0.0")]
#![rustc_const_stable(feature = "stability_attribute_implies", since = "1.0.0")]

#[stable(feature = "stability_attribute_implies", since = "1.0.0")]
#[rustc_const_stable(feature = "const_foo", since = "1.62.0")]
pub const fn foo() {}

#[stable(feature = "stability_attribute_implies", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_foobar", issue = "1", implied_by = "const_foo")]
pub const fn foobar() {}
