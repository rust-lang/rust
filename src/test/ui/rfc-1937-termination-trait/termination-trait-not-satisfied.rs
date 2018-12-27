struct ReturnType {}

fn main() -> ReturnType { //~ ERROR `main` has invalid return type `ReturnType`
    ReturnType {}
}
