#![feature(postfix_match)]

fn main() {
    1 as i32.match {};
    //~^ ERROR cast cannot be followed by a postfix match
    //~| ERROR non-exhaustive patterns
}
