//! Test that `Box` cannot be used with a lifetime argument.
//! regression test for issue <https://github.com/rust-lang/rust/issues/18423>

struct Foo<'a> {
    x: Box<'a, isize>,
    //~^ ERROR struct takes 0 lifetime arguments but 1 lifetime argument was supplied
}

fn main() {}
