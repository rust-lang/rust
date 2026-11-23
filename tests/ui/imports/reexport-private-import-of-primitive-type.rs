// Regression test for https://github.com/rust-lang/rust/issues/165827

//@ edition: 2024

use str;
pub use str as b;
//~^ ERROR `str` is only public within the crate, and cannot be re-exported outside

fn main() {}
