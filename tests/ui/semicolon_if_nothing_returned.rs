#![warn(clippy::semicolon_if_nothing_returned)]

fn get_unit() {}

// the functions below trigger the lint
fn main() {
    println!("Hello")
}

fn hello() {
    get_unit()
}

// this is fine
fn print_sum(a: i32, b: i32) {
    println!("{}", a + b);
    assert_eq!(true, false);
}
