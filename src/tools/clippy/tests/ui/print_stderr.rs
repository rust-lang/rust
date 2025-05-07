#![warn(clippy::print_stderr)]

fn main() {
    eprintln!("Hello");
    //~^ print_stderr

    println!("This should not do anything");
    eprint!("World");
    //~^ print_stderr

    print!("Nor should this");
}
