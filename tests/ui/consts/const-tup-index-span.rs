// Test spans of errors

const TUP: (usize,) = 5usize << 64;
//~^ ERROR mismatched types
//~| expected `(usize,)`, found `usize`
const ARR: [i32; TUP.0] = [];
//~^ constant

fn main() {
}
