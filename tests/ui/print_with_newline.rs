#![feature(plugin)]
#![plugin(clippy)]
#![warn(print_with_newline)]

fn main() {
    print!("Hello\n");
    print!("Hello {}\n", "world");
    print!("Hello {} {}\n\n", "world", "#2");
    print!("{}\n", 1265);

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
