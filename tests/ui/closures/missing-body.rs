// Checks that the compiler complains about the missing closure body and does not
// crash.
// This is a regression test for <https://github.com/rust-lang/rust/issues/143128>.

fn main() { |b: [str; _]| {}; }
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for closures
//~| ERROR the size for values of type `str` cannot be known at compilation time
