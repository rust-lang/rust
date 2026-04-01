//! auxiliary crate for <https://github.com/rust-lang/rust/issues/11680>

enum Foo {
    Bar(isize)
}

pub mod test {
    enum Foo {
        Bar(isize)
    }
}
