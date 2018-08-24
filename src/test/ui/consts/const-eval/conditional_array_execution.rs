#![warn(const_err)]

const X: u32 = 5;
const Y: u32 = 6;
const FOO: u32 = [X - Y, Y - X][(X < Y) as usize];
//~^ WARN this constant cannot be used

fn main() {
    println!("{}", FOO);
    //~^ ERROR erroneous constant used
    //~| E0080
}
