//@run-rustfix

#![allow(unused)]
#![warn(clippy::read_line_without_trim)]

fn main() {
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    input.pop();
    let _x: i32 = input.parse().unwrap(); // don't trigger here, newline character is popped

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let _x: i32 = input.parse().unwrap();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let _x = input.parse::<i32>().unwrap();
}
