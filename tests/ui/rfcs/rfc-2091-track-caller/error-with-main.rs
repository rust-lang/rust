#[track_caller] //~ ERROR `main` function is not allowed to be
fn main() {
    panic!("{}: oh no", std::panic::Location::caller());
}
