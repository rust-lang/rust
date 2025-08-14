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
    //~^ read_line_without_trim

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let _x = input.parse::<i32>().unwrap();
    //~^ read_line_without_trim

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let _x = input.parse::<u32>().unwrap();
    //~^ read_line_without_trim

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let _x = input.parse::<f32>().unwrap();
    //~^ read_line_without_trim

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let _x = input.parse::<bool>().unwrap();
    //~^ read_line_without_trim

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    // this is actually ok, so don't lint here
    let _x = input.parse::<String>().unwrap();

    // comparing with string literals
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    if input == "foo" {
        //~^ read_line_without_trim
        println!("This will never ever execute!");
    }

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    if input.ends_with("foo") {
        //~^ read_line_without_trim
        println!("Neither will this");
    }
}
