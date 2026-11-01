//@ edition:2024
// Regression tests for https://github.com/rust-lang/rust/issues/164044

const V: Vec<()> = Vec::new();

macro_rules! generated_plain {
    () => {
        if let V = Vec::new() {} //~ ERROR constant of non-structural type
    };
}

macro_rules! generated_nested {
    () => {
        if let V = vec![] {} //~ ERROR constant of non-structural type
    };
}

fn main() {
    if let V = Vec::new() {} //~ ERROR constant of non-structural type
    if let V = vec![] {} //~ ERROR constant of non-structural type
    generated_plain!();
    generated_nested!();
}
