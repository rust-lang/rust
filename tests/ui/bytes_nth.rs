// run-rustfix

#![warn(clippy::bytes_nth)]

fn main() {
    let _ = "Hello".bytes().nth(3);

    let _ = String::from("Hello").bytes().nth(3);
}
