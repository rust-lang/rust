// Regression test for https://github.com/rust-lang/rust/issues/122463
struct Foo {
    a: Vec<(
        /// Docstring
        //~^ ERROR doc comments cannot be applied to types
        f32,
        f32,
    )>,
}

fn main() {}
