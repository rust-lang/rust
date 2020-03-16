// Test spans of errors

const TUP: (usize,) = 5usize << 64;
//~^ ERROR mismatched types
//~| ERROR mismatched types
const ARR: [i32; TUP.0] = [];
//~^ ERROR evaluation of constant value failed

fn main() {
}
