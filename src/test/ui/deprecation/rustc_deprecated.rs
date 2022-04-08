// compile-flags: --crate-type=lib

#![feature(staged_api)]
#![stable(since = "1.0.0", feature = "rust1")]

#[rustc_deprecated( //~ ERROR `#[rustc_deprecated]` has been removed
    //~^ HELP use `#[deprecated]` instead
    since = "1.100.0",
    reason = "text" //~ ERROR `reason` has been renamed
    //~^ HELP use `note` instead
)]
#[stable(feature = "rust1", since = "1.0.0")]
fn foo() {}
