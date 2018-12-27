#![warn(const_err)]

const X: u32 = 5;
const Y: u32 = 6;
const FOO: u32 = [X - Y, Y - X][(X < Y) as usize];
//~^ WARN any use of this value will cause an error

fn main() {
    println!("{}", FOO);
    //~^ ERROR
}
