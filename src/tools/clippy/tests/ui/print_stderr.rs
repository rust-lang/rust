#![warn(clippy::print_stderr)]

fn main() {
    eprintln!("Hello");
    //~^ ERROR: use of `eprintln!`
    //~| NOTE: `-D clippy::print-stderr` implied by `-D warnings`
    println!("This should not do anything");
    eprint!("World");
    //~^ ERROR: use of `eprint!`
    print!("Nor should this");
}
