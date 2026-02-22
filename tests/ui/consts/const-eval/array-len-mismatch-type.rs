//! Regression test for <https://github.com/rust-lang/rust/issues/133966>
pub struct Data([[&'static str]; 5_i32]);
//~^ ERROR mismatched types
const _: &'static Data = unsafe { &*(&[] as *const Data) };
fn main() {}
