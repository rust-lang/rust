// https://github.com/rust-lang/rust/issues/46377
#![crate_name="foo"]

//@ has 'foo/index.html' '//dd' 'Check out this struct!'
/// # Check out this struct!
pub struct SomeStruct;
