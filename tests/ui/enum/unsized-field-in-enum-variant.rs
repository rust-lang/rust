//! Regression test for <https://github.com/rust-lang/rust/issues/35988>
enum E {
    V([Box<E>]),
    //~^ ERROR the size for values of type
}

fn main() {}
