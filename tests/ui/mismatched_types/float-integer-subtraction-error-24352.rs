// https://github.com/rust-lang/rust/issues/24352
fn main() {
    1.0f64 - 1.0;
    1.0f64 - 1 //~ ERROR E0277
}
