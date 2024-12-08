//@ run-rustfix
use std::io::stdin;

fn get_name() -> String {
    let mut your_name = String::new();
    stdin()
        .read_line(&mut your_name)
        .expect("Failed to read the line for some reason");
    your_name.trim() //~ ERROR E0308
}

fn main() {
    println!("Hello, What is your name? ");
    let your_name = get_name();
    println!("Hello, {}", your_name)
}
