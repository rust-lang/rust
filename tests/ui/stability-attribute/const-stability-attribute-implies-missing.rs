#![crate_type = "lib"]
#![feature(staged_api)]
#![stable(feature = "stability_attribute_implies", since = "1.0.0")]
#![rustc_const_stable(feature = "stability_attribute_implies", since = "1.0.0")]

// Tests that `implied_by = "const_bar"` results in an error being emitted if `const_bar` does not
// exist.

#[stable(feature = "stability_attribute_implies", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_foobar", issue = "1", implied_by = "const_bar")]
//~^ ERROR feature `const_bar` implying `const_foobar` does not exist
pub const fn foobar() -> u32 {
    0
}

const VAR: u32 = foobar();
