// Test spans of errors

const TUP: (usize,) = 5usize << 64;
//~^ ERROR mismatched types
//~| NOTE expected `(usize,)`, found `usize`
//~| NOTE expected tuple `(usize,)`
const ARR: [i32; TUP.0] = [];

fn main() {
}
