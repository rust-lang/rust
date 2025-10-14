#![core::contracts::requires]
//~^ ERROR use of unstable library feature `contracts`
//~| ERROR inner macro attributes are unstable
//~| ERROR wrong meta list delimiters
//~| ERROR `#[prelude_import]` is for use by rustc only
//~| ERROR mismatched types
#[allow{}]
fn main() {}
