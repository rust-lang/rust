#[track_caller] //~ ERROR `main` function is not allowed to be
fn main() {
    panic!("{}: oh no", std::panic::Location::caller());
}

//@ reference: attributes.codegen.track_caller.allowed-positions
