// This is a regression test for <https://github.com/rust-lang/rust/issues/152682>
struct Foo<'a>(<& /*'a*/ [fn()] as core::ops::Deref>::Target); // adding the lifetime solves the ice
             //~^ ERROR: missing lifetime specifier [E0106]
const _: *const Foo = 0 as _;
                        //~^ ERROR: cannot cast `i32` to a pointer that is wide [E0606]
fn main() {}
