#![warn(clippy::print_stderr)]

fn main() {
    eprintln!("Hello");
    println!("This should not do anything");
    eprint!("World");
    print!("Nor should this");
}
