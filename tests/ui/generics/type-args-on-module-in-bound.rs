//! regression test for <https://github.com/rust-lang/rust/issues/22706>
fn is_copy<T: ::std::marker<i32>::Copy>() {}
//~^ ERROR type arguments are not allowed on module `marker` [E0109]
fn main() {}
