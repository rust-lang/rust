// https://github.com/rust-lang/rust/issues/27862
#![crate_name="issue_27862"]

/// Tests  | Table
/// ------|-------------
/// t = b | id = \|x\| x
pub struct Foo; //@ has issue_27862/type.Foo.html //td 'id = |x| x'
