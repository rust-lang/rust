// Ensures our "remove semicolon" suggestion isn't hardcoded with a character width,
// in case it was accidentally mixed up with a greek question mark.
// issue: rust-lang/rust#123607

pub fn square(num: i32) -> i32 {
    //~^ ERROR mismatched types
    num * num;
    //~^ ERROR unknown start of token
}

fn main() {}
