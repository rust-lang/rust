enum bottom { }

fn main() {
    let x = &() as *const () as *const bottom;
    match x { } //~ ERROR non-exhaustive patterns
}
