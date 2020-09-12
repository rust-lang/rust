// Tests that the error message uses the word Copy, not Pod.

fn check_bound<T:Copy>(_: T) {}

fn main() {
    check_bound("nocopy".to_string()); //~ ERROR : Copy` is not satisfied
}
