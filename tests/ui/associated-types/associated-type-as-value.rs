//! regression test for <https://github.com/rust-lang/rust/issues/38919>

fn foo<T: Iterator>() {
    T::Item; //~ ERROR no associated item named `Item` found
}

fn main() { }
