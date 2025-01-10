#![warn(clippy::primitive_method_to_numeric_cast)]

fn main() {
    let _ = u16::max as usize; //~ primitive_method_to_numeric_cast
}
