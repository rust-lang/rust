// Test spans of errors

const TUP: (usize,) = 5usize << 64;
//~^ ERROR mismatched types
//~| NOTE_NONVIRAL expected `(usize,)`, found `usize`
const ARR: [i32; TUP.0] = [];

fn main() {
}
