#[derive(Debug)]
struct Demo {
    #[skip] //~ ERROR the `#[skip]` attribute is experimental
    f1: (),
}

fn main() {}
