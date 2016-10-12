#![feature(plugin)]
#![plugin(clippy)]
#![deny(print_with_newline)]

fn main() {
    print!("Hello\n"); //~ERROR using `print!()` with a format string
    print!("Hello {}\n", "world"); //~ERROR using `print!()` with a format string
    print!("Hello {} {}\n\n", "world", "#2"); //~ERROR using `print!()` with a format string
    print!("{}\n", 1265); //~ERROR using `print!()` with a format string

    // these are all fine
    print!("");
    print!("Hello");
    println!("Hello");
    println!("Hello\n");
    println!("Hello {}\n", "world");
    print!("Issue\n{}", 1265);
    print!("{}", 1265);
    print!("\n{}", 1275);
}
