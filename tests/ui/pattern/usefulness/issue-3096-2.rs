enum Bottom { }

fn main() {
    let x = &() as *const () as *const Bottom;
    match x { } //~ ERROR non-exhaustive patterns
}
