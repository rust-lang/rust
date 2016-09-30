#![feature(plugin)]
#![plugin(clippy)]
#![deny(print_with_newline)]

fn main() {
    print!("Hello");
    print!("Hello\n"); //~ERROR using `print!()` with a format string
    print!("Hello {}\n", "world"); //~ERROR using `print!()` with a format string
    print!("Hello {} {}\n\n", "world", "#2"); //~ERROR using `print!()` with a format string

    // these are all fine
    println!("Hello");
    println!("Hello\n");
    println!("Hello {}\n", "world");
}
