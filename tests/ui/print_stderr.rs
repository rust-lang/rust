#![warn(clippy::print_stderr)]

fn main() {
    eprintln!("Hello");
    eprint!("World");
}
