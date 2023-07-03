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

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let _x = input.parse::<u32>().unwrap();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let _x = input.parse::<f32>().unwrap();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let _x = input.parse::<bool>().unwrap();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    // this is actually ok, so don't lint here
    let _x = input.parse::<String>().unwrap();
}
