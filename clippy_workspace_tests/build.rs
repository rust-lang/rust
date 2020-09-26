#![deny(clippy::print_stdout)]

fn main() {
    // Test for #6041
    println!("Hello");
    print!("Hello");
}
