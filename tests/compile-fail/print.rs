#![feature(plugin)]
#![plugin(clippy)]

#[deny(print_stdout)]

fn main() {
    println!("Hello"); //~ERROR use of `println!`
    print!("Hello"); //~ERROR use of `print!`

    vec![1, 2];
}
