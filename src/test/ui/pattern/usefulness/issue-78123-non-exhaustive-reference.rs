enum A {}

fn f(a: &A) {
    match a {} //~ ERROR non-exhaustive patterns: type `&A` is non-empty
}

fn main() {}
