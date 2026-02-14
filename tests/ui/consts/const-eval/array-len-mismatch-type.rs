//! Regression test for <https://github.com/rust-lang/rust/issues/133966>
pub struct Data([[&'static str]; 5_i32]);
//~^ ERROR the constant `5` is not of type `usize`
//~| ERROR the size for values of type `[&'static str]` cannot be known at compilation time
//~| ERROR mismatched types
const _: &'static Data = unsafe { &*(&[] as *const Data) };
//~^ ERROR the type `[[&str]; 5]` has an unknown layout
fn main() {}
