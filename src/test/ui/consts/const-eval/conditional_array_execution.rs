// build-fail

#![warn(const_err)]

const X: u32 = 5;
const Y: u32 = 6;
const FOO: u32 = [X - Y, Y - X][(X < Y) as usize];
//~^ WARN any use of this value will cause an error
//~| WARN this was previously accepted by the compiler but is being phased out

fn main() {
    println!("{}", FOO);
    //~^ ERROR evaluation of constant value failed
    //~| WARN erroneous constant used [const_err]
    //~| WARN this was previously accepted by the compiler but is being phased out
}
