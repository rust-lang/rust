#![allow(clippy::print_literal)]
#![warn(clippy::print_with_newline)]

fn main() {
    print!("Hello\n");
    print!("Hello {}\n", "world");
    print!("Hello {} {}\n", "world", "#2");
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
    print!("\n\n");
    print!("like eof\n\n");
    print!("Hello {} {}\n\n", "world", "#2");
    println!("\ndon't\nwarn\nfor\nmultiple\nnewlines\n"); // #3126
    println!("\nbla\n\n"); // #3126
}
