struct Foo {
    a: Vec<(
        /// Docstring
        //~^ ERROR expected type, found doc comment
        f32,
        f32,
    )>,
}

fn main() {}
