#![warn(clippy::bytes_count_to_len)]

fn main() {
    let s1 = String::from("world");

    //test warning against a string literal
    "hello".bytes().count();

    //test warning against a string variable
    s1.bytes().count();

    //make sure using count() normally doesn't trigger warning
    let vector = [0, 1, 2];
    let size = vector.iter().count();
}
